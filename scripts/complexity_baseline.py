"""Measure baseline component runtime as sample size increases."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import random
import time

import numpy as np
import pandas as pd

from scenarios import make_scenarios
from src.dgps import generate_sar_data
from src.methods import DPQMLE, WangSongGaussian
from src.simulation import stable_seed


N_VALUES = [400, 600, 800, 1000, 1200]
REPEATS = 2


def main():
    rows = []

    template = next(
        s for s in make_scenarios()
        if s["epsilon"] == 1.0
    )

    for N in N_VALUES:
        settings = {
            **template,
            "N": N,
            "delta": 10 / (N ** 1.1),
        }

        for rep in range(REPEATS):
            data_seed = stable_seed(
                1234, N, rep, "data"
            )

            start = time.perf_counter()
            data = generate_sar_data(
                settings,
                data_seed,
            )
            dgp_time = time.perf_counter() - start

            for method in [
                DPQMLE(),
                WangSongGaussian(),
            ]:
                method_seed = stable_seed(
                    1234,
                    N,
                    rep,
                    method.name,
                )

                np.random.seed(method_seed)
                random.seed(method_seed)

                start = time.perf_counter()
                method.fit(data, settings)
                method_time = (
                    time.perf_counter() - start
                )

                rows.append(
                    {
                        "N": N,
                        "rep": rep,
                        "component": method.name,
                        "seconds": method_time,
                    }
                )

            rows.append(
                {
                    "N": N,
                    "rep": rep,
                    "component": "DGP",
                    "seconds": dgp_time,
                }
            )

        print(f"Completed N={N}")

    result = pd.DataFrame(rows)

    result.to_csv(
        "results/performance/"
        "baseline_complexity.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
