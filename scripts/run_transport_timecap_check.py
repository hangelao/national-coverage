"""Corrected fixed-July C2/C3 coverage check with a true 120-km cap."""
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

COVERAGE = (0.50, 0.75, 0.85, 1.00)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs" / "transport_timecap_check")
    args = parser.parse_args()
    assert_fixed_july()
    locations, solar, cold, states = load_fixed_july_inputs()
    cfg = fixed_july_config(max_transport_hours=3.0)
    assert cfg.van_speed_kmh * cfg.max_transport_link_hours == 120.0
    run_coverage_frontier(
        states, solar, cold, locations, SNAPSHOTS, WEIGHTS, cfg,
        str(args.output_dir), configs=("C2", "C3"),
        coverage_levels=COVERAGE, supply="solar")

if __name__ == "__main__":
    main()
