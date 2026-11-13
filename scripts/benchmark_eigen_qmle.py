"""Benchmark pre-eigenvalue versus eigenvalue DP-QMLE."""

from pathlib import Path
import sys
import random
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.optimization2.function_qmle_dp import (
    qmle_dp_estimate as qmle_pre_eigen,
)

from src.function_qmle_dp import (
    qmle_dp_estimate as qmle_eigen,
)

from src.generation import generate_compare


OUTPUT = (
    ROOT
    / "results"
    / "performance"
    / "eigen_qmle_benchmark.csv"
)

N_VALUES = [400, 600, 800, 1000, 1200]

# Use 5 for final reporting.
REPEATS = 5

P = 2
BETA = np.array(
    [[0.3], [0.3]]
)
RHO = 0.2
EPSILON = 1.0
NETWORK = "PowerLaw"


def time_method(
    method,
    X,
    Y,
    W,
    epsilon,
    delta,
    noise_seed,
):
    """Time one DP-QMLE fit with a fixed DP random stream."""

    np.random.seed(noise_seed)
    random.seed(noise_seed)

    start = time.perf_counter()

    estimate, *_ = method(
        X.copy(),
        Y.copy(),
        W.copy(),
        epsilon,
        delta,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return np.asarray(estimate), elapsed


def main():
    rows = []

    for N in N_VALUES:

        delta = 10 / (N ** 1.1)

        for rep in range(REPEATS):

            # Generate one identical dataset for both implementations.
            data_seed = (
                1234
                + 10000 * N
                + rep
            )

            np.random.seed(data_seed)
            random.seed(data_seed)

            X, Y, W = generate_compare(
                N,
                P,
                BETA,
                RHO,
                NETWORK,
            )

            # Use exactly the same DP noise stream for both fits.
            noise_seed = (
                9999
                + 10000 * N
                + rep
            )

            # Alternate execution order to reduce systematic
            # first/second-run timing bias.
            if rep % 2 == 0:
                first_name = "pre_eigen"
                first_method = qmle_pre_eigen

                second_name = "eigen"
                second_method = qmle_eigen

            else:
                first_name = "eigen"
                first_method = qmle_eigen

                second_name = "pre_eigen"
                second_method = qmle_pre_eigen

            first_estimate, first_time = time_method(
                first_method,
                X,
                Y,
                W,
                EPSILON,
                delta,
                noise_seed,
            )

            second_estimate, second_time = time_method(
                second_method,
                X,
                Y,
                W,
                EPSILON,
                delta,
                noise_seed,
            )

            # Map results back to implementation names.
            times = {
                first_name: first_time,
                second_name: second_time,
            }

            estimates = {
                first_name: first_estimate,
                second_name: second_estimate,
            }

            # Verify that the optimization preserves the estimator.
            np.testing.assert_allclose(
                estimates["eigen"],
                estimates["pre_eigen"],
                rtol=1e-6,
                atol=1e-8,
            )

            pre_time = times["pre_eigen"]
            eigen_time = times["eigen"]

            rows.append(
                {
                    "N": N,
                    "rep": rep,
                    "pre_eigen_seconds": pre_time,
                    "eigen_seconds": eigen_time,
                    "speedup": (
                        pre_time
                        / eigen_time
                    ),
                }
            )

            print(
                f"N={N}, rep={rep + 1}/{REPEATS}: "
                f"pre={pre_time:.4f}s, "
                f"eigen={eigen_time:.4f}s, "
                f"speedup={pre_time / eigen_time:.2f}x"
            )

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
            pre_eigen_mean=(
                "pre_eigen_seconds",
                "mean",
            ),
            pre_eigen_median=(
                "pre_eigen_seconds",
                "median",
            ),
            eigen_mean=(
                "eigen_seconds",
                "mean",
            ),
            eigen_median=(
                "eigen_seconds",
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
    print("Benchmark summary:")
    print(summary)


if __name__ == "__main__":
    main()
