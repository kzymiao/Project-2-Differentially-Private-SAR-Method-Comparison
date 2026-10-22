"""Simulation orchestration for the DP-SAR method comparison."""

import hashlib
import json
import random
import time

import numpy as np
import pandas as pd

from .dgps import generate_sar_data
from .methods import PRIMARY_METHODS
from .metrics import parameter_rows


def stable_seed(master_seed, N, rep, label):
    """Derive an order-independent 32-bit seed from simulation identifiers.

    Epsilon is intentionally omitted. Thus, for a fixed N and replicate, both
    privacy budgets use the same generated data. Method-specific random streams
    are also paired across epsilon values, which reduces Monte Carlo noise in
    privacy-budget comparisons.
    """
    payload = json.dumps([master_seed, N, rep, label], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def run_simulation(scenarios, n_reps=50, seed=1234, methods=None):
    """Run all scenarios and return replicate-level parameter results."""
    methods = PRIMARY_METHODS if methods is None else methods
    rows = []
    total = len(scenarios) * n_reps
    completed = 0

    for settings in scenarios:
        truth = np.asarray([*settings["beta"], settings["rho"]], dtype=float)

        for rep in range(n_reps):
            data_seed = stable_seed(seed, settings["N"], rep, "data")
            data = generate_sar_data(settings, data_seed)

            for method in methods:
                method_seed = stable_seed(seed, settings["N"], rep, method.name)
                np.random.seed(method_seed)
                random.seed(method_seed)

                start = time.perf_counter()
                status, message = "ok", ""

                try:
                    estimate = method.fit(data, settings)
                    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
                        raise ValueError("non-finite estimate or unexpected parameter shape")
                except Exception as exc:
                    estimate = np.full_like(truth, np.nan, dtype=float)
                    status = "failed"
                    message = f"{type(exc).__name__}: {exc}"

                elapsed = time.perf_counter() - start

                metadata = {
                    "N": settings["N"],
                    "p": settings["p"],
                    "rho_true": settings["rho"],
                    "network": settings["network"],
                    "epsilon": settings["epsilon"],
                    "delta": settings["delta"],
                    "rep": rep,
                    "method": method.name,
                    "status": status,
                    "message": message,
                    "elapsed_seconds": elapsed,
                }
                rows.extend(parameter_rows(estimate, truth, metadata))

            completed += 1
            print(f"\rCompleted {completed}/{total} datasets", end="", flush=True)

    print()
    return pd.DataFrame(rows)
