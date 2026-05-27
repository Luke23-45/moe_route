#!/usr/bin/env python
"""Phase C runner — FineWeb Baseline Comparison.

The final multi-seed main-result section of the project.  Runs the main
methods on FineWeb ``sample-10BT`` for the paper's scale-up results.

Per the publication plan:
  - Main comparison: Reflected v2 vs DeepSeek-LFB (2 seeds each)
  - Top-2 anchor run (1 seed)
  - ``capacity_factor ∈ {0.5, 1.0}``
  - If budget is tight, preserve both capacities for Reflected v2 and
    DeepSeek-LFB first, then reduce Top-2 anchor coverage.
  - Evaluate: held-out PPL + zero-shot suite (HellaSwag, PIQA, WinoGrande,
    ARC-Easy, ARC-Challenge)
  - Report compute-normalized results: active params, tokens, wall-clock,
    tokens/sec, routing overhead.

Usage:
  python -m studies.runners.run_phase_c
  python -m studies.runners.run_phase_c --dry-run
  python -m studies.runners.run_phase_c --only reflected_sparse
  python -m studies.runners.run_phase_c --seeds 1337
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from studies.utils.experiment_spec import PHASE_C_SPECS, Phase
from studies.utils.runner_base import CONFIG_ROOT, PhaseRunner
from studies.utils.validation import validate_study

STUDY_NAME = "phase_c_fineweb_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase C: FineWeb baseline comparison.\n"
            "Reflected v2 vs DeepSeek-LFB (2 seeds) + Top-2 anchor (1 seed) "
            "at cf={0.5,1.0}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    all_names = sorted({s.name for s in PHASE_C_SPECS})
    parser.add_argument(
        "--only",
        choices=all_names,
        action="append",
        default=[],
        help="Run only selected method(s). Can be repeated.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list override (e.g. '1337,42').",
    )
    parser.add_argument(
        "--run-prefix",
        default="phase_c",
        help="Prefix for run names.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/checkpoints/phase_c",
        help="Root directory for checkpoints.",
    )
    parser.add_argument(
        "--result-root",
        default="artifacts/results",
        help="Root directory for result persistence.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="GPUs per node for distributed training.",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse seed override
    seed_override = None
    if args.seeds:
        seed_override = [int(s.strip()) for s in args.seeds.split(",")]

    # Build runner
    runner = PhaseRunner(
        study_name=STUDY_NAME,
        phase=Phase.C,
        specs=PHASE_C_SPECS,
        result_root=args.result_root,
        run_prefix=args.run_prefix,
    )

    # Build schedule
    runs = runner.build_schedule(
        only=args.only or None,
        seed_override=seed_override,
    )

    # Validate
    report = validate_study(runs, CONFIG_ROOT, STUDY_NAME)
    report.print_report()
    report.raise_if_failed()

    # Print plan
    runner.print_plan(runs)

    # Prepare data (FineWeb)
    if not args.skip_prepare:
        runner.prepare_data("fineweb_10bt", rebuild=args.rebuild_data, dry_run=args.dry_run)

    # Execute
    results = runner.execute(
        runs,
        checkpoint_root=args.checkpoint_root,
        common_overrides=args.set,
        nproc_per_node=args.nproc_per_node,
        skip_prepare=not args.skip_prepare,
        dry_run=args.dry_run,
    )

    # Report
    runner.aggregate_and_summarize(runs)
    runner.print_results_table(runs)

    # Exit status
    failed = [r for r, ok in results if not ok]
    if failed:
        print(f"\n[{STUDY_NAME}] {len(failed)} run(s) FAILED:")
        for run, _ in [(r, ok) for r, ok in results if not ok]:
            print(f"  ✗ {run.run_name}")
        raise SystemExit(1)

    print(f"\n[{STUDY_NAME}] All {len(results)} run(s) completed successfully.\n")


if __name__ == "__main__":
    main()
