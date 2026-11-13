import numpy as np
import random
import pandas as pd
from benchmarks.baseline.function_qmle_dp import (
    qmle_dp_estimate as baseline_qmle_dp_estimate,
)

from src.function_qmle_dp import (
    qmle_dp_estimate as optimized_qmle_dp_estimate,
)

from src.generation import generate_compare



def test_linear_solve_matches_explicit_inverse():
    rng = np.random.default_rng(123)

    N = 30
    rho = 0.2

    # Construct a fixed row-normalized weight matrix.
    W = rng.uniform(size=(N, N))
    np.fill_diagonal(W, 0.0)
    W = W / W.sum(axis=1, keepdims=True)

    S = np.eye(N) - rho * W

    rhs = rng.normal(size=(N, 1))

    # Unit 2 implementation
    y_baseline = np.linalg.inv(S).dot(rhs)

    # Unit 3 optimized implementation
    y_optimized = np.linalg.solve(S, rhs)

    np.testing.assert_allclose(
        y_optimized,
        y_baseline,
        rtol=1e-8,
        atol=1e-10,
    )


def test_trace_identity_matches_full_matrix_product():
    rng = np.random.default_rng(123)

    N = 30
    rho = 0.2

    W = rng.uniform(
        size=(N, N)
    )

    S = (
        np.eye(N)
        - rho * W
    )

    S_inv = np.linalg.inv(S)

    baseline = np.trace(
        W.dot(S_inv)
    )

    optimized = np.sum(
        W * S_inv.T
    )

    np.testing.assert_allclose(
        optimized,
        baseline,
        rtol=1e-12,
        atol=1e-12,
    )


def test_optimized_qmle_matches_baseline():
    N = 80
    p = 2

    beta = np.array(
        [[0.3], [0.3]]
    )

    rho = 0.2

    np.random.seed(123)
    random.seed(123)

    X, Y, W = generate_compare(
        N,
        p,
        beta,
        rho,
        "PowerLaw",
    )

    epsilon = 1.0
    delta = 10 / (N ** 1.1)

    # Use exactly the same DP noise stream.
    np.random.seed(999)
    random.seed(999)

    baseline, *_ = (
        baseline_qmle_dp_estimate(
            X.copy(),
            Y.copy(),
            W.copy(),
            epsilon,
            delta,
        )
    )

    np.random.seed(999)
    random.seed(999)

    optimized, *_ = (
        optimized_qmle_dp_estimate(
            X.copy(),
            Y.copy(),
            W.copy(),
            epsilon,
            delta,
        )
    )

    np.testing.assert_allclose(
        optimized,
        baseline,
        rtol=1e-6,
        atol=1e-8,
    )


def test_simulation_reuses_data_across_epsilon(monkeypatch):
    from src import simulation

    calls = []

    def fake_generate_sar_data(settings, seed):
        calls.append(
            (
                settings["N"],
                seed,
            )
        )

        N = settings["N"]
        p = settings["p"]

        return {
            "X": np.zeros((N, p)),
            "Y": np.zeros((N, 1)),
            "W": np.eye(N),
            "beta_true": np.asarray(
                settings["beta"]
            ),
            "rho_true": settings["rho"],
        }

    class DummyMethod:
        name = "dummy"

        def fit(self, data, settings):
            return np.asarray(
                [
                    *settings["beta"],
                    settings["rho"],
                ],
                dtype=float,
            )

    monkeypatch.setattr(
        simulation,
        "generate_sar_data",
        fake_generate_sar_data,
    )

    scenarios = [
        {
            "N": 10,
            "p": 2,
            "beta": [0.3, 0.3],
            "rho": 0.2,
            "network": "PowerLaw",
            "epsilon": 1.0,
            "delta": 0.01,
        },
        {
            "N": 10,
            "p": 2,
            "beta": [0.3, 0.3],
            "rho": 0.2,
            "network": "PowerLaw",
            "epsilon": 2.0,
            "delta": 0.01,
        },
    ]

    result = simulation.run_simulation(
        scenarios,
        n_reps=3,
        seed=1234,
        methods=[DummyMethod()],
    )

    # Two epsilon scenarios × three reps would previously
    # generate six datasets.
    #
    # The optimized implementation should generate only
    # three datasets: one per replicate.
    assert len(calls) == 3

    assert sorted(
        result["epsilon"].unique()
    ) == [1.0, 2.0]

    assert len(result) == (
        2  # epsilon values
        * 3  # replicates
        * 3  # parameters
    )

def test_eigenvalue_trace_matches_inverse_trace():
    rng = np.random.default_rng(123)

    N = 30
    rho = 0.2

    W = rng.uniform(
        size=(N, N)
    )

    W /= W.sum(
        axis=1,
        keepdims=True,
    )

    S = (
        np.eye(N)
        - rho * W
    )

    # Current Optimization-2 calculation.
    S_inv = np.linalg.inv(S)

    baseline = np.sum(
        W * S_inv.T
    )

    # Proposed eigenvalue calculation.
    eigenvalues = np.linalg.eigvals(W)

    optimized = np.sum(
        eigenvalues
        / (
            1.0
            - rho * eigenvalues
        )
    )

    np.testing.assert_allclose(
        optimized.real,
        baseline,
        rtol=1e-9,
        atol=1e-10,
    )

    assert abs(optimized.imag) < 1e-9


def test_eigenvalue_trace_across_rho_values():
    rng = np.random.default_rng(456)

    N = 30

    W = rng.uniform(
        size=(N, N)
    )

    W /= W.sum(
        axis=1,
        keepdims=True,
    )

    eigenvalues = np.linalg.eigvals(W)

    for rho in [
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
    ]:
        S = (
            np.eye(N)
            - rho * W
        )

        S_inv = np.linalg.inv(S)

        baseline = np.sum(
            W * S_inv.T
        )

        optimized = np.sum(
            eigenvalues
            / (
                1.0
                - rho * eigenvalues
            )
        )

        np.testing.assert_allclose(
            optimized.real,
            baseline,
            rtol=1e-8,
            atol=1e-9,
        )



import random
import numpy as np

from benchmarks.optimization2.function_qmle_dp import (
    qmle_dp_estimate as qmle_before_eigen,
)
from src.function_qmle_dp import (
    qmle_dp_estimate as qmle_after_eigen,
)
from src.generation import generate_compare


def test_eigenvalue_qmle_matches_previous_qmle():
    N = 80
    p = 2
    beta = np.array([[0.3], [0.3]])
    rho = 0.2

    np.random.seed(123)
    random.seed(123)

    X, Y, W = generate_compare(
        N,
        p,
        beta,
        rho,
        "PowerLaw",
    )

    epsilon = 1.0
    delta = 10 / (N ** 1.1)

    # Same DP noise stream.
    np.random.seed(999)
    random.seed(999)

    before, *_ = qmle_before_eigen(
        X.copy(),
        Y.copy(),
        W.copy(),
        epsilon,
        delta,
    )

    np.random.seed(999)
    random.seed(999)

    after, *_ = qmle_after_eigen(
        X.copy(),
        Y.copy(),
        W.copy(),
        epsilon,
        delta,
    )

    np.testing.assert_allclose(
        after,
        before,
        rtol=1e-6,
        atol=1e-8,
    )

def test_parallel_matches_serial():
    from scenarios import make_scenarios
    from src.simulation import (
        run_simulation,
        run_simulation_parallel,
    )

    # Only use N=400 to keep the test lightweight.
    scenarios = [
        s
        for s in make_scenarios()
        if s["N"] == 400
    ]

    serial = run_simulation(
        scenarios,
        n_reps=1,
        seed=1234,
    )

    parallel = run_simulation_parallel(
        scenarios,
        n_reps=1,
        seed=1234,
        workers=2,
    )

    sort_cols = [
        "N",
        "epsilon",
        "rep",
        "method",
        "parameter",
    ]

    serial = (
        serial
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    parallel = (
        parallel
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    compare_cols = [
        c
        for c in serial.columns
        if c != "elapsed_seconds"
    ]

    pd.testing.assert_frame_equal(
        serial[compare_cols],
        parallel[compare_cols],
        check_exact=False,
        rtol=1e-7,
        atol=1e-9,
    )

