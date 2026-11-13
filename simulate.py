"""Run the raw Monte Carlo simulation and save replicate-level results."""

import argparse
from pathlib import Path

from scenarios import make_scenarios
from src.simulation import (
    run_simulation,
    run_simulation_parallel,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/raw/runs.csv"),
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run independent Monte Carlo replicates in parallel.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes used with --parallel.",
    )

    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = make_scenarios()

    if args.parallel:
        result = run_simulation_parallel(
            scenarios,
            n_reps=args.reps,
            seed=args.seed,
            workers=args.workers,
        )
    else:
        result = run_simulation(
            scenarios,
            n_reps=args.reps,
            seed=args.seed,
        )

    result.to_csv(
        args.out,
        index=False,
    )

    mode = (
        f"parallel ({args.workers} workers)"
        if args.parallel
        else "serial"
    )

    print(
        f"Saved {len(result):,} parameter-level rows "
        f"to {args.out} using {mode} execution."
    )


if __name__ == "__main__":
    main()