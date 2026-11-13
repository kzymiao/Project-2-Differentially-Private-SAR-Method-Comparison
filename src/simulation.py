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
from concurrent.futures import ProcessPoolExecutor

def stable_seed(master_seed, N, rep, label):
    """Derive an order-independent 32-bit seed from simulation identifiers.

    Epsilon is intentionally omitted. Thus, for a fixed N and replicate, both
    privacy budgets use the same generated data. Method-specific random streams
    are also paired across epsilon values, which reduces Monte Carlo noise in
    privacy-budget comparisons.
    """
    payload = json.dumps([master_seed, N, rep, label], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

def dgp_key(settings):
    """Return the settings that uniquely determine the generated dataset."""
    return (
        settings["N"],
        settings["p"],
        tuple(settings["beta"]),
        settings["rho"],
        settings["network"],
    )


def group_scenarios_by_dgp(scenarios):
    """Group scenarios that share exactly the same data-generating settings."""
    groups = {}

    for settings in scenarios:
        key = dgp_key(settings)

        if key not in groups:
            groups[key] = []

        groups[key].append(settings)

    return list(groups.values())



def run_one_dataset(
    scenario_group,
    rep,
    seed,
    methods=None,
):
    """Run all privacy settings and methods for one DGP replicate."""

    if methods is None:
        methods = PRIMARY_METHODS

    base_settings = scenario_group[0]

    truth = np.asarray(
        [
            *base_settings["beta"],
            base_settings["rho"],
        ],
        dtype=float,
    )

    data_seed = stable_seed(
        seed,
        base_settings["N"],
        rep,
        "data",
    )

    data = generate_sar_data(
        base_settings,
        data_seed,
    )

    rows = []

    for settings in scenario_group:
        for method in methods:

            method_seed = stable_seed(
                seed,
                settings["N"],
                rep,
                method.name,
            )

            np.random.seed(method_seed)
            random.seed(method_seed)

            start = time.perf_counter()

            status = "ok"
            message = ""

            try:
                estimate = method.fit(
                    data,
                    settings,
                )

                if (
                    estimate.shape != truth.shape
                    or not np.all(np.isfinite(estimate))
                ):
                    raise ValueError(
                        "non-finite estimate or "
                        "unexpected parameter shape"
                    )

            except Exception as exc:
                estimate = np.full_like(
                    truth,
                    np.nan,
                    dtype=float,
                )

                status = "failed"
                message = (
                    f"{type(exc).__name__}: {exc}"
                )

            elapsed = (
                time.perf_counter()
                - start
            )

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

            rows.extend(
                parameter_rows(
                    estimate,
                    truth,
                    metadata,
                )
            )

    return rows  


def run_simulation(
    scenarios,
    n_reps=50,
    seed=1234,
    methods=None,
):
    """Run simulation serially."""

    scenario_groups = group_scenarios_by_dgp(
        scenarios
    )

    rows = []

    total = (
        len(scenario_groups)
        * n_reps
    )

    completed = 0

    for scenario_group in scenario_groups:

        for rep in range(n_reps):

            rows.extend(
                run_one_dataset(
                    scenario_group,
                    rep,
                    seed,
                    methods=methods,
                )
            )

            completed += 1

            print(
                f"\rCompleted "
                f"{completed}/{total} datasets",
                end="",
                flush=True,
            )

    print()

    return pd.DataFrame(rows)

def _parallel_worker(task):
    scenario_group, rep, seed = task

    return run_one_dataset(
        scenario_group,
        rep,
        seed,
        methods=None,
    )

def run_simulation_parallel(
    scenarios,
    n_reps=50,
    seed=1234,
    workers=4,
):
    """Run independent DGP replicates in separate processes."""

    scenario_groups = group_scenarios_by_dgp(
        scenarios
    )

    tasks = [
        (
            scenario_group,
            rep,
            seed,
        )
        for scenario_group in scenario_groups
        for rep in range(n_reps)
    ]

    rows = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        results = executor.map(
            _parallel_worker,
            tasks,
        )

        for completed, result_rows in enumerate(
            results,
            start=1,
        ):
            rows.extend(result_rows)

            print(
                f"\rCompleted "
                f"{completed}/{len(tasks)} datasets",
                end="",
                flush=True,
            )

    print()

    return pd.DataFrame(rows)