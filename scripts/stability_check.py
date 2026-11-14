"""Summarize numerical failures and extreme estimation errors."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "raw"
    / "runs.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "performance"
)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            "Run the simulation first."
        )

    raw = pd.read_csv(INPUT)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # One row per fitted model rather than one row per parameter.
    fits = (
        raw[
            [
                "N",
                "epsilon",
                "rep",
                "method",
                "status",
                "message",
            ]
        ]
        .drop_duplicates()
    )

    fit_summary = (
        fits
        .groupby(
            [
                "N",
                "epsilon",
                "method",
            ]
        )
        .agg(
            n_fits=("status", "size"),
            n_failed=(
                "status",
                lambda x: int(
                    (x != "ok").sum()
                ),
            ),
        )
        .reset_index()
    )

    fit_summary.to_csv(
        OUTPUT_DIR
        / "stability_fit_summary.csv",
        index=False,
    )

    raw["abs_error"] = np.abs(
        raw["error"]
    )

    error_summary = (
        raw
        .groupby(
            [
                "N",
                "epsilon",
                "method",
                "parameter",
            ]
        )
        .agg(
            n=("error", "count"),
            max_abs_error=(
                "abs_error",
                "max",
            ),
            p99_abs_error=(
                "abs_error",
                lambda x: x.quantile(0.99),
            ),
        )
        .reset_index()
    )

    error_summary.to_csv(
        OUTPUT_DIR
        / "stability_error_summary.csv",
        index=False,
    )

    print("\nFit failures:")
    print(fit_summary)

    print("\nLargest absolute errors:")
    print(
        raw[
            [
                "N",
                "epsilon",
                "rep",
                "method",
                "parameter",
                "error",
            ]
        ]
        .assign(
            abs_error=lambda x: np.abs(
                x["error"]
            )
        )
        .sort_values(
            "abs_error",
            ascending=False,
        )
        .head(10)
    )


if __name__ == "__main__":
    main()
