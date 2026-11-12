"""Profile one representative Unit 2 simulation condition."""

import cProfile

import pstats

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scenarios import make_scenarios
from src.simulation import run_simulation


PROFILE_DIR = Path("results/performance")
PROFILE_FILE = PROFILE_DIR / "baseline.prof"
PROFILE_TEXT = PROFILE_DIR / "baseline_profile.txt"


def workload():
    scenarios = make_scenarios()

    scenario = next(
        s for s in scenarios
        if s["N"] == 800
        and s["epsilon"] == 1.0
    )

    run_simulation(
        [scenario],
        n_reps=1,
        seed=1234,
    )


def main():
    PROFILE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiler = cProfile.Profile()

    # Imports have already happened before profiling starts.
    profiler.enable()
    workload()
    profiler.disable()

    profiler.dump_stats(PROFILE_FILE)

    with PROFILE_TEXT.open("w") as stream:
        stats = pstats.Stats(
            profiler,
            stream=stream,
        )

        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(30)

    print(
        f"Saved profile to {PROFILE_FILE}"
    )
    print(
        f"Saved summary to {PROFILE_TEXT}"
    )


if __name__ == "__main__":
    main()