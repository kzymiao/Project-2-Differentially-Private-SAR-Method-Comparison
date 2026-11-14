"""Benchmark original Unit 2 DP-QMLE against final optimized DP-QMLE."""

from pathlib import Path
import sys
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.baseline.function_qmle_dp import (
    qmle_dp_estimate as baseline_qmle,
)

from src.function_qmle_dp import (
    qmle_dp_estimate as optimized_qmle,
)

from src.generation import generate_compare


N_VALUES = [400, 600, 800, 1000, 1200]
REPEATS = 5

P = 2
BETA = np.array([[0.3], [0.3]])
RHO = 0.2
EPSILON = 1.0
NETWORK = "PowerLaw"

OUTPUT = (
    ROOT
    / "results"
    / "performance"
    / "final_benchmark.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "results"
    / "performance"
    / "final_benchmark_summary.csv"
)

FIGURE_DIR = (
    ROOT
    / "results"
    / "figures"
)


def time_method(
    method,
    X,
    Y,
    W,
    epsilon,
    delta,
    seed,
):
    np.random.seed(seed)
    random.seed(seed)

    start = time.perf_counter()

    estimate, *_ = method(
        X.copy(),
        Y.copy(),
        W.copy(),
        epsilon,
        delta,
    )

    elapsed = time.perf_counter() - start

    return np.asarray(estimate), elapsed


def main():
    rows = []

    for N in N_VALUES:
        delta = 10 / (N ** 1.1)

        for rep in range(REPEATS):
            data_seed = 1234 + 10000 * N + rep

            np.random.seed(data_seed)
            random.seed(data_seed)

            X, Y, W = generate_compare(
                N,
                P,
                BETA,
                RHO,
                NETWORK,
            )

            noise_seed = 9999 + 10000 * N + rep

            # Alternate order to reduce first/second-run bias.
            if rep % 2 == 0:
                methods = [
                    ("baseline", baseline_qmle),
                    ("optimized", optimized_qmle),
                ]
            else:
                methods = [
                    ("optimized", optimized_qmle),
                    ("baseline", baseline_qmle),
                ]

            times = {}
            estimates = {}

            for name, method in methods:
                estimate, elapsed = time_method(
                    method,
                    X,
                    Y,
                    W,
                    EPSILON,
                    delta,
                    noise_seed,
                )

                times[name] = elapsed
                estimates[name] = estimate

            np.testing.assert_allclose(
                estimates["optimized"],
                estimates["baseline"],
                rtol=1e-6,
                atol=1e-8,
            )

            rows.append(
                {
                    "N": N,
                    "rep": rep,
                    "baseline_seconds": times["baseline"],
                    "optimized_seconds": times["optimized"],
                    "speedup": (
                        times["baseline"]
                        / times["optimized"]
                    ),
                }
            )

            print(
                f"N={N}, rep={rep + 1}/{REPEATS}: "
                f"baseline={times['baseline']:.4f}s, "
                f"optimized={times['optimized']:.4f}s, "
                f"speedup="
                f"{times['baseline']/times['optimized']:.2f}x"
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
            baseline_median=(
                "baseline_seconds",
                "median",
            ),
            optimized_median=(
                "optimized_seconds",
                "median",
            ),
            speedup_median=(
                "speedup",
                "median",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(summary)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Runtime vs N
    fig, ax = plt.subplots()

    ax.plot(
        summary["N"],
        summary["baseline_median"],
        marker="o",
        label="Unit 2 baseline",
    )

    ax.plot(
        summary["N"],
        summary["optimized_median"],
        marker="o",
        label="Final optimized",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Median DP-QMLE runtime (seconds)")
    ax.set_title("Baseline vs optimized DP-QMLE runtime")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "runtime_vs_N.png",
        dpi=300,
    )

    fig.savefig(
        FIGURE_DIR / "runtime_vs_N.pdf",
    )

    plt.close(fig)

    # Speedup vs N
    fig, ax = plt.subplots()

    ax.plot(
        summary["N"],
        summary["speedup_median"],
        marker="o",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Speedup over Unit 2 baseline")
    ax.set_title("DP-QMLE optimization speedup")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "speedup_vs_N.png",
        dpi=300,
    )

    fig.savefig(
        FIGURE_DIR / "speedup_vs_N.pdf",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
