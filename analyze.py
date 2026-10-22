"""Summarize simulation output and create comparison figures."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import summarize


PARAMETERS = ["beta1", "beta2", "rho"]
METHOD_ORDER = ["DP-QMLE (ours)", "Wang-Song Gaussian FM"]


def _ordered_methods(frame):
    """Return methods in the preferred display order when available."""
    present = list(frame["method"].drop_duplicates())
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(method for method in present if method not in ordered)
    return ordered


def _epsilon_values(frame):
    """Return sorted privacy-budget values present in a result table."""
    return sorted(float(x) for x in frame["epsilon"].dropna().unique())


def make_bias_figure(summary):
    """Plot absolute bias versus N, faceted by epsilon and parameter.

    Bias is nonnegative by definition: abs(mean(theta_hat - theta)). Approximate
    +/-2 MCSE bars use the MCSE of the underlying mean error; the lower whisker
    is clipped at zero so the display respects the nonnegative bias scale.
    """
    epsilons = _epsilon_values(summary)
    methods = _ordered_methods(summary)

    fig, axes = plt.subplots(
        len(epsilons),
        len(PARAMETERS),
        figsize=(12.5, 3.1 * len(epsilons)),
        sharex=True,
        squeeze=False,
        constrained_layout=False,
    )

    for row, epsilon in enumerate(epsilons):
        for col, parameter in enumerate(PARAMETERS):
            ax = axes[row, col]
            panel = summary[
                (summary["epsilon"] == epsilon)
                & (summary["parameter"] == parameter)
            ]

            for method in methods:
                method_data = panel[panel["method"] == method].sort_values("N")
                if method_data.empty:
                    continue

                y = method_data["bias"].to_numpy(float)
                mcse = method_data["bias_mcse"].to_numpy(float)
                uncertainty = 2 * mcse
                lower = np.minimum(uncertainty, y)
                yerr = np.vstack([lower, uncertainty])

                ax.errorbar(
                    method_data["N"],
                    y,
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    label=method,
                )

            if row == 0:
                ax.set_title(parameter)
            if col == 0:
                ax.set_ylabel(f"epsilon = {epsilon:g}\nAbsolute bias")
            if row == len(epsilons) - 1:
                ax.set_xlabel("Sample size N")

            # Keeps small biases visible when a comparator occasionally has a
            # much larger error, while preserving zero on the axis.
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25, which="both")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def make_rmse_figure(summary):
    """Plot RMSE versus N, faceted by epsilon and parameter."""
    epsilons = _epsilon_values(summary)
    methods = _ordered_methods(summary)

    fig, axes = plt.subplots(
        len(epsilons),
        len(PARAMETERS),
        figsize=(12.5, 3.1 * len(epsilons)),
        sharex=True,
        squeeze=False,
        constrained_layout=False,
    )

    for row, epsilon in enumerate(epsilons):
        for col, parameter in enumerate(PARAMETERS):
            ax = axes[row, col]
            panel = summary[
                (summary["epsilon"] == epsilon)
                & (summary["parameter"] == parameter)
            ]

            for method in methods:
                method_data = panel[panel["method"] == method].sort_values("N")
                if method_data.empty:
                    continue

                y = method_data["rmse"].to_numpy(float)
                mcse = method_data["rmse_mcse"].to_numpy(float)
                uncertainty = 2 * mcse

                # A log axis requires positive lower endpoints. This clips only
                # the displayed lower whisker, not the estimated RMSE itself.
                lower = np.minimum(uncertainty, 0.95 * y)
                yerr = np.vstack([lower, uncertainty])

                ax.errorbar(
                    method_data["N"],
                    y,
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    label=method,
                )

            if row == 0:
                ax.set_title(parameter)
            if col == 0:
                ax.set_ylabel(f"epsilon = {epsilon:g}\nRMSE")
            if row == len(epsilons) - 1:
                ax.set_xlabel("Sample size N")

            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def make_epsilon_rmse_figure(summary):
    """Directly compare RMSE at epsilon=1 and epsilon=2 for each method."""
    epsilons = _epsilon_values(summary)
    methods = _ordered_methods(summary)

    fig, axes = plt.subplots(
        len(methods),
        len(PARAMETERS),
        figsize=(12.5, 3.1 * len(methods)),
        sharex=True,
        squeeze=False,
        constrained_layout=False,
    )

    for row, method in enumerate(methods):
        for col, parameter in enumerate(PARAMETERS):
            ax = axes[row, col]
            panel = summary[
                (summary["method"] == method)
                & (summary["parameter"] == parameter)
            ]

            for epsilon in epsilons:
                data = panel[panel["epsilon"] == epsilon].sort_values("N")
                if data.empty:
                    continue

                y = data["rmse"].to_numpy(float)
                mcse = data["rmse_mcse"].to_numpy(float)
                uncertainty = 2 * mcse
                lower = np.minimum(uncertainty, 0.95 * y)

                ax.errorbar(
                    data["N"],
                    y,
                    yerr=np.vstack([lower, uncertainty]),
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    label=f"epsilon = {epsilon:g}",
                )

            if row == 0:
                ax.set_title(parameter)
            if col == 0:
                ax.set_ylabel(f"{method}\nRMSE")
            if row == len(methods) - 1:
                ax.set_xlabel("Sample size N")

            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(labels)),
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def make_error_diagnostic(raw):
    """Boxplots of estimation errors at the largest N and epsilon=1."""
    largest_n = int(raw["N"].max())
    epsilon = min(float(x) for x in raw["epsilon"].dropna().unique())
    data = raw[
        (raw["N"] == largest_n)
        & (raw["epsilon"] == epsilon)
        & (raw["status"] == "ok")
    ]
    methods = _ordered_methods(data)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 3.6),
        constrained_layout=False,
    )

    for ax, parameter in zip(axes, PARAMETERS):
        panel = data[data["parameter"] == parameter]
        values = [
            panel.loc[panel["method"] == method, "error"].dropna().to_numpy()
            for method in methods
        ]

        ax.boxplot(values, tick_labels=methods, showfliers=True)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(parameter)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15, labelsize=9)

    axes[0].set_ylabel("Estimation error")
    fig.suptitle(
        f"Estimation-error distributions at N={largest_n}, epsilon={epsilon:g}",
        y=1.04,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/raw/runs.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--stage", choices=["analyze", "figures", "all"], default="all")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)

    if args.stage in {"analyze", "all"}:
        summary = summarize(raw)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary, index=False)
        print(f"Saved summary to {args.summary}")
    else:
        summary = pd.read_csv(args.summary)

    if args.stage in {"figures", "all"}:
        if not args.summary.exists():
            summary = summarize(raw)
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(args.summary, index=False)
        else:
            summary = pd.read_csv(args.summary)

        args.figures.mkdir(parents=True, exist_ok=True)

        figures = {
            "absolute_bias_vs_N": make_bias_figure(summary),
            "rmse_vs_N": make_rmse_figure(summary),
            "epsilon_rmse_comparison": make_epsilon_rmse_figure(summary),
            "error_distribution_diagnostic": make_error_diagnostic(raw),
        }

        for name, fig in figures.items():
            fig.savefig(args.figures / f"{name}.png", dpi=300, bbox_inches="tight")
            fig.savefig(args.figures / f"{name}.pdf", bbox_inches="tight")
            plt.close(fig)

        print(f"Saved figures to {args.figures}")


if __name__ == "__main__":
    main()
