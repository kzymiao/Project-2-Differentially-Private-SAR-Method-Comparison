"""Fast checks for the simulation study."""

import numpy as np
import pandas as pd

from scenarios import make_scenarios
from src.dgps import generate_sar_data
from src.metrics import summarize


def small_settings():
    return {
        "N": 80,
        "p": 2,
        "beta": [0.3, 0.3],
        "rho": 0.2,
        "network": "Dyad",
        "epsilon": 1.0,
        "delta": 0.05,
    }


def test_final_design_crosses_sample_size_and_epsilon():
    scenarios = make_scenarios()
    assert {s["N"] for s in scenarios} == {400, 600, 800, 1000, 1200}
    assert {s["epsilon"] for s in scenarios} == {1.0, 2.0}
    assert len(scenarios) == 10


def test_dgp_shapes_bounds_and_weight_rows():
    data = generate_sar_data(small_settings(), seed=17)
    assert data["X"].shape == (80, 2)
    assert data["Y"].shape == (80, 1)
    assert np.all(data["X"] >= -1) and np.all(data["X"] <= 1)
    row_sums = np.asarray(data["W"].sum(axis=1)).ravel()
    assert np.allclose(row_sums[row_sums > 0], 1.0)


def test_dgp_reproducibility():
    first = generate_sar_data(small_settings(), seed=123)
    second = generate_sar_data(small_settings(), seed=123)
    np.testing.assert_allclose(first["X"], second["X"])
    np.testing.assert_allclose(first["Y"], second["Y"])
    np.testing.assert_allclose(first["W"].toarray(), second["W"].toarray())


def test_metric_summary_uses_absolute_bias_and_separates_epsilon():
    raw = pd.DataFrame(
        {
            "N": [100] * 6,
            "epsilon": [1.0] * 3 + [2.0] * 3,
            "method": ["m"] * 6,
            "parameter": ["rho"] * 6,
            "status": ["ok"] * 6,
            "error": [1.0, -1.0, 2.0, -2.0, -2.0, 1.0],
        }
    )

    out = summarize(raw).sort_values("epsilon").reset_index(drop=True)

    assert len(out) == 2
    assert np.isclose(out.loc[0, "bias"], abs((1 - 1 + 2) / 3))
    assert np.isclose(out.loc[0, "rmse"], np.sqrt((1 + 1 + 4) / 3))
    assert np.isclose(out.loc[1, "bias"], abs((-2 - 2 + 1) / 3))
    assert (out["bias"] >= 0).all()
    assert (out["rmse"] >= 0).all()
