"""Performance measures and Monte Carlo uncertainty calculations."""

import numpy as np
import pandas as pd


PARAMETER_NAMES = ["beta1", "beta2", "rho"]


def parameter_rows(estimate, truth, metadata):
    """Return one raw result row per model parameter."""
    rows = []
    for name, est, true in zip(PARAMETER_NAMES, estimate, truth):
        error = float(est - true)
        rows.append(
            {
                **metadata,
                "parameter": name,
                "estimate": float(est),
                "truth": float(true),
                "error": error,
                "squared_error": error ** 2,
            }
        )
    return rows


def summarize(raw):
    """Summarize absolute bias, RMSE, and Monte Carlo uncertainty.

    Bias is defined exactly as in the original comparison script:

        abs(mean(theta_hat - theta)).

    ``bias_mcse`` is the Monte Carlo standard error of the underlying mean
    estimation error. It is used only as an approximate uncertainty bar around
    absolute bias; plotting code clips the lower whisker at zero.
    """
    groups = ["N", "epsilon", "method", "parameter"]
    rows = []

    for keys, group in raw.groupby(groups, sort=True):
        good = group.loc[group["status"] == "ok"].dropna(subset=["error"])
        B = len(good)

        base = {
            **dict(zip(groups, keys)),
            "n_ok": B,
            "n_total": len(group),
        }

        if B == 0:
            rows.append(
                {
                    **base,
                    "bias": np.nan,
                    "bias_mcse": np.nan,
                    "rmse": np.nan,
                    "rmse_mcse": np.nan,
                }
            )
            continue

        errors = good["error"].to_numpy(float)
        squared_errors = errors ** 2

        bias = abs(errors.mean())
        bias_mcse = errors.std(ddof=1) / np.sqrt(B) if B > 1 else np.nan

        mse = squared_errors.mean()
        rmse = np.sqrt(mse)
        mse_mcse = (
            squared_errors.std(ddof=1) / np.sqrt(B)
            if B > 1
            else np.nan
        )
        rmse_mcse = (
            mse_mcse / (2 * rmse)
            if B > 1 and rmse > 0
            else np.nan
        )

        rows.append(
            {
                **base,
                "bias": bias,
                "bias_mcse": bias_mcse,
                "rmse": rmse,
                "rmse_mcse": rmse_mcse,
            }
        )

    return pd.DataFrame(rows)
