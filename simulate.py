"""Run the raw Monte Carlo simulation and save replicate-level results."""

import argparse
from pathlib import Path

from scenarios import make_scenarios
from src.simulation import run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", type=Path, default=Path("results/raw/runs.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run_simulation(make_scenarios(), n_reps=args.reps, seed=args.seed)
    result.to_csv(args.out, index=False)
    print(f"Saved {len(result):,} parameter-level rows to {args.out}")


if __name__ == "__main__":
    main()
