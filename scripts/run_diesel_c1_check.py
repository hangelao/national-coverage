"""Corrected coarse C1 diesel cases at 25%, 75%, and 100% only."""
from __future__ import annotations
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path
from budget_frontier import run_coverage_frontier
from corrected_fixed_july_common import (ROOT, SNAPSHOTS, WEIGHTS,
    assert_fixed_july, fixed_july_config, load_fixed_july_inputs)

COVERAGE = (0.25, 0.75, 1.00)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "corrected_coarse_c1_diesel_fixed_july")
    args = parser.parse_args()
    assert_fixed_july()
    locations, solar, cold, states = load_fixed_july_inputs()
    run_coverage_frontier(
        states, solar, cold, locations, SNAPSHOTS, WEIGHTS,
        fixed_july_config(), str(args.output_dir), configs=("C1",),
        coverage_levels=COVERAGE, supply="diesel")

if __name__ == "__main__":
    main()
