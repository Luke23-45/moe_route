"""Robust routing-stress evaluation suite launcher.

Evaluates the best and latest checkpoints for one or more TinyStories runs and writes
JSON/Markdown summaries with routing-stress metrics under held-out evaluation data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from _runner import module_cmd, run_command, validate_override_syntax

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StressSpec:
    name: str
    description: str


SUITE: tuple[StressSpec, ...] = (
    StressSpec(name="dense", description="Dense baseline"),
    StressSpec(name="top1", description="Top-1 sparse routing baseline"),
    StressSpec(name="top2", description="Top-2 sparse routing baseline"),
    StressSpec(name="reflected", description="Dense reflected controller"),
    StressSpec(name="reflected_sparse", description="Sparse reflected controller"),
)


def validate_spec(
    spec: StressSpec,
    *,
    checkpoint_root: Path,
    validation_root: Path | None,
    run_prefix: str,
) -> list[str]:
    errors: list[str] = []
    run_name = f"{run_prefix}_{spec.name}"
    checkpoint_dir = checkpoint_root / run_name
    if not checkpoint_dir.exists():
        errors.append(f"[{spec.name}] checkpoint directory not found: {checkpoint_dir}")
    elif not any(checkpoint_dir.glob("step_*.pt")):
        errors.append(f"[{spec.name}] no checkpoints found under: {checkpoint_dir}")

    if validation_root is not None:
        metrics_csv = validation_root / run_name / "metrics.csv"
        if not metrics_csv.exists():
            errors.append(f"[{spec.name}] validation metrics CSV not found: {metrics_csv}")
    return errors


def validate_suite(
    specs: list[StressSpec],
    *,
    checkpoint_root: Path,
    validation_root: Path | None,
    run_prefix: str,
) -> None:
    errors: list[str] = []
    for spec in specs:
        errors.extend(
            validate_spec(
                spec,
                checkpoint_root=checkpoint_root,
                validation_root=validation_root,
                run_prefix=run_prefix,
            )
        )
    if errors:
        print(f"\n{'='*60}")
        print("  ROUTING-STRESS VALIDATION FAILED")
        print(f"{'='*60}")
        for err in errors:
            print(f"  X {err}")
        print(f"{'='*60}\n")
        raise SystemExit(1)


def build_overrides(
    *,
    checkpoint_dir: Path,
    validation_metrics_csv: Path | None,
    output_json: Path,
    output_md: Path,
    max_batches: int | None,
    eval_split: str | None,
) -> list[str]:
    overrides = [
        f"checkpoint_dir={checkpoint_dir}",
        f"output_file={output_json}",
        f"output_md={output_md}",
    ]
    if validation_metrics_csv is not None:
        overrides.append(f"validation_metrics_csv={validation_metrics_csv}")
    if max_batches is not None:
        overrides.append(f"max_batches={max_batches}")
    if eval_split is not None:
        overrides.append(f"eval_split={eval_split}")
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run routing-stress evaluation across selected TinyStories runs."
    )
    parser.add_argument(
        "--only",
        choices=[spec.name for spec in SUITE],
        action="append",
        default=[],
        help="Evaluate only selected run(s). Can be provided multiple times.",
    )
    parser.add_argument("--run-prefix", default="tinystories")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints/tinystories")
    parser.add_argument(
        "--validation-root",
        default="artifacts/validation_results",
        help="Root directory containing per-run metrics.csv files. Use --skip-validation-check to allow missing CSVs.",
    )
    parser.add_argument("--output-dir", default="artifacts/routing_stress")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument(
        "--skip-validation-check",
        action="store_true",
        help="Allow missing validation metrics.csv and fall back to latest checkpoint as best.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [spec for spec in SUITE if not args.only or spec.name in args.only]
    checkpoint_root = Path(args.checkpoint_root)
    validation_root = None if args.skip_validation_check else Path(args.validation_root)
    output_root = Path(args.output_dir)

    print(f"\n{'='*70}")
    print(f"  Routing Stress Suite - {len(selected)} run(s)")
    print(f"{'='*70}")
    for spec in selected:
        run_name = f"{args.run_prefix}_{spec.name}"
        print(f"  {spec.name:>16}  |  run={run_name:<28}  {spec.description}")
    print(f"{'='*70}\n")

    validate_suite(
        selected,
        checkpoint_root=checkpoint_root,
        validation_root=validation_root,
        run_prefix=args.run_prefix,
    )
    print("  [OK] Checkpoint and validation inputs look consistent.\n")

    for idx, spec in enumerate(selected, 1):
        run_name = f"{args.run_prefix}_{spec.name}"
        checkpoint_dir = checkpoint_root / run_name
        validation_csv = None if validation_root is None else validation_root / run_name / "metrics.csv"
        output_json = output_root / f"{run_name}.json"
        output_md = output_root / f"{run_name}.md"

        print(f"[stress] ({idx}/{len(selected)}) Evaluating: {run_name}")
        overrides = build_overrides(
            checkpoint_dir=checkpoint_dir,
            validation_metrics_csv=validation_csv,
            output_json=output_json,
            output_md=output_md,
            max_batches=args.max_batches,
            eval_split=args.eval_split,
        )
        validate_override_syntax(overrides)
        code = run_command(module_cmd("moe_route.cli.routing_stress", overrides), args.dry_run)
        if code != 0:
            print(f"\n[stress] ERROR: Routing-stress evaluation for '{run_name}' failed with exit code {code}.")
            raise SystemExit(code)
        print(f"[stress] OK: Wrote {output_json} and {output_md}")

    print(f"\n[stress] All {len(selected)} routing-stress evaluation(s) completed successfully.\n")


if __name__ == "__main__":
    main()
