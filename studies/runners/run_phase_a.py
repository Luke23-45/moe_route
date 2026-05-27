#!/usr/bin/env python
"""Phase A runner — TinyStories Baseline Comparison.

Runs the three main MoE methods (Top-2, DeepSeek-LFB, Reflected v2)
plus two supporting controls (Dense, Top-1) on TinyStories.

Per the publication plan:
  - Main methods run at ``capacity_factor ∈ {0.5, 1.0}``
  - Main methods run with **3 seeds** each
  - Controls run with **1 seed** only
  - Required outputs: validation loss/PPL, top-1 accuracy, router CV/Gini,
    drop rate, entropy, throughput, early-vs-late loss, loss tails.

Usage:
  python -m studies.runners.run_phase_a
  python -m studies.runners.run_phase_a --dry-run
  python -m studies.runners.run_phase_a --smoke
  python -m studies.runners.run_phase_a --only top2 --only reflected_sparse
  python -m studies.runners.run_phase_a --seeds 1337,42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path for studies package imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from studies.utils.experiment_spec import PHASE_A_SPECS, Phase
from studies.utils.runner_base import CONFIG_ROOT, PhaseRunner
from studies.utils.validation import validate_study

STUDY_NAME = "phase_a_tinystories_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A: TinyStories baseline comparison.\n"
            "Runs Top-2, DeepSeek-LFB, and Reflected v2 at cf={0.5,1.0} "
            "with 3 seeds each, plus Dense and Top-1 controls."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    all_names = sorted({s.name for s in PHASE_A_SPECS})
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
        default="phase_a",
        help="Prefix for run names.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/checkpoints/phase_a",
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
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data preparation.")
    parser.add_argument("--rebuild-data", action="store_true", help="Force rebuild data cache.")
    parser.add_argument("--smoke", action="store_true", help="Smoke test (CPU, few steps).")
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
        phase=Phase.A,
        specs=PHASE_A_SPECS,
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

    # Prepare data
    if not args.skip_prepare:
        data_config = "tinystories_smoke" if args.smoke else "tinystories"
        runner.prepare_data(data_config, rebuild=args.rebuild_data, dry_run=args.dry_run)

    # Execute
    results = runner.execute(
        runs,
        checkpoint_root=args.checkpoint_root,
        common_overrides=args.set,
        nproc_per_node=args.nproc_per_node,
        skip_prepare=not args.skip_prepare,
        dry_run=args.dry_run,
        smoke=args.smoke,
    )

    # Report
    if not args.dry_run:
        runner.aggregate_and_summarize(runs)
        runner.print_results_table(runs)

    # Exit status
    failed = [r for r, ok in results if not ok]
    if failed:
        print(f"\n[{STUDY_NAME}] {len(failed)} run(s) FAILED:")
        for run, _ in [(r, ok) for r, ok in results if not ok]:
            print(f"  ✗ {run.run_name}")
        raise SystemExit(1)

    n = len(results)
    print(f"\n[{STUDY_NAME}] All {n} run(s) completed successfully.\n")


if __name__ == "__main__":
    main()
