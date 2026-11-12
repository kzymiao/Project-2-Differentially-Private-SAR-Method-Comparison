"""Benchmark explicit inverse versus direct linear solve."""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "results" / "performance" / "dgp_solve_benchmark.csv"

N_VALUES = [400, 600, 800, 1000, 1200]
REPEATS = 5


def main():
    rng = np.random.default_rng(1234)

    rows = []

    for N in N_VALUES:
        # Create a stable row-normalized matrix for timing.
        W = rng.uniform(size=(N, N))
        np.fill_diagonal(W, 0.0)
        W /= W.sum(axis=1, keepdims=True)

        rho = 0.2

        S = np.eye(N) - rho * W

        rhs = rng.normal(
            size=(N, 1)
        )

        for rep in range(REPEATS):
            # Baseline: explicitly construct S^{-1}.
            start = time.perf_counter()

            y_inverse = (
                np.linalg.inv(S)
                .dot(rhs)
            )

            inverse_seconds = (
                time.perf_counter()
                - start
            )

            # Optimized: solve S y = rhs directly.
            start = time.perf_counter()

            y_solve = np.linalg.solve(
                S,
                rhs,
            )

            solve_seconds = (
                time.perf_counter()
                - start
            )

            # Verify that the two implementations agree.
            np.testing.assert_allclose(
                y_inverse,
                y_solve,
                rtol=1e-9,
                atol=1e-11,
            )

            rows.append(
                {
                    "N": N,
                    "rep": rep,
                    "inverse_seconds": inverse_seconds,
                    "solve_seconds": solve_seconds,
                    "speedup": (
                        inverse_seconds
                        / solve_seconds
                    ),
                }
            )

        print(f"Completed N={N}")

    result = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    summary = (
        result
        .groupby("N")
        .agg(
            inverse_mean=(
                "inverse_seconds",
                "mean",
            ),
            inverse_median=(
                "inverse_seconds",
                "median",
            ),
            solve_mean=(
                "solve_seconds",
                "mean",
            ),
            solve_median=(
                "solve_seconds",
                "median",
            ),
            speedup_mean=(
                "speedup",
                "mean",
            ),
            speedup_median=(
                "speedup",
                "median",
            ),
        )
    )

    print()
    print(summary)


if __name__ == "__main__":
    main()
