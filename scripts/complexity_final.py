"""Estimate empirical runtime scaling from final benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "performance"
    / "final_benchmark_summary.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "performance"
    / "complexity_summary.csv"
)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            "Run `make benchmark` first."
        )

    data = pd.read_csv(INPUT)

    log_n = np.log(
        data["N"].to_numpy(float)
    )

    rows = []

    for label, column in [
        ("Unit 2 baseline", "baseline_median"),
        ("Final optimized", "optimized_median"),
    ]:
        log_time = np.log(
            data[column].to_numpy(float)
        )

        slope, intercept = np.polyfit(
            log_n,
            log_time,
            1,
        )

        rows.append(
            {
                "implementation": label,
                "empirical_exponent": slope,
                "intercept": intercept,
            }
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

    print(result)


if __name__ == "__main__":
    main()
