"""Robust TinyStories experiment suite launcher.

Prepares data once, then runs a validated set of experiments (dense, top1, top2, reflected).
Each experiment is defined as a self-documenting ExperimentSpec with pre-flight validation
that catches config mismatches, missing files, and compatibility errors before any training.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from _runner import module_cmd, run_command, validate_override_syntax

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Experiment Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    """Self-documenting experiment specification with validation metadata.

    Attributes:
        name:           CLI key used with --only (e.g. "dense", "reflected").
        experiment:     Hydra experiment config name (without .yaml).
        model:          Hydra model config name (without .yaml).
        router:         Hydra router config name (None for dense baseline).
        is_moe:         Whether this experiment uses MoE layers.
        dispatch_mode:  "sparse" (top-k), "dense" (soft), or "none" (standard FFN).
        description:    Human-readable description shown in --help and plan printout.
    """

    name: str
    experiment: str
    model: str
    router: str | None = None
    is_moe: bool = False
    dispatch_mode: str = "none"
    description: str = ""


SUITE: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="dense",
        experiment="tinystories_dense",
        model="tiny_dense",
        router=None,
        is_moe=False,
        dispatch_mode="none",
        description="Dense baseline (standard FFN, no MoE)",
    ),
    ExperimentSpec(
        name="top1",
        experiment="tinystories_top1",
        model="tiny_moe",
        router="top1",
        is_moe=True,
        dispatch_mode="sparse",
        description="Top-1 sparse MoE routing",
    ),
    ExperimentSpec(
        name="top2",
        experiment="tinystories_top2",
        model="tiny_moe",
        router="top2",
        is_moe=True,
        dispatch_mode="sparse",
        description="Top-2 sparse MoE routing",
    ),
    ExperimentSpec(
        name="reflected",
        experiment="tinystories_reflected_v2",
        model="tiny_moe",
        router="reflected",
        is_moe=True,
        dispatch_mode="dense",
        description="Reflected controller (dense soft routing, aux-loss-free)",
    ),
)

SUITE_BY_NAME: dict[str, ExperimentSpec] = {spec.name: spec for spec in SUITE}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(spec: ExperimentSpec, config_root: Path) -> list[str]:
    """Return a list of error messages for a single spec. Empty list = valid."""
    errors: list[str] = []

    # ── Config file existence checks ──
    exp_path = config_root / "experiment" / f"{spec.experiment}.yaml"
    if not exp_path.exists():
        errors.append(f"[{spec.name}] Experiment config not found: {exp_path.relative_to(REPO_ROOT)}")

    model_path = config_root / "model" / f"{spec.model}.yaml"
    if not model_path.exists():
        errors.append(f"[{spec.name}] Model config not found: {model_path.relative_to(REPO_ROOT)}")

    if spec.router is not None:
        router_path = config_root / "router" / f"{spec.router}.yaml"
        if not router_path.exists():
            errors.append(f"[{spec.name}] Router config not found: {router_path.relative_to(REPO_ROOT)}")

    # ── Compatibility constraints ──
    if spec.is_moe and spec.router is None:
        errors.append(f"[{spec.name}] MoE experiment requires a router config, but router is None")

    if not spec.is_moe and spec.router is not None:
        errors.append(f"[{spec.name}] Dense experiment should not specify a router (got '{spec.router}')")

    if not spec.is_moe and spec.dispatch_mode != "none":
        errors.append(
            f"[{spec.name}] Dense experiment dispatch_mode must be 'none', got '{spec.dispatch_mode}'"
        )

    if spec.is_moe and spec.dispatch_mode == "none":
        errors.append(
            f"[{spec.name}] MoE experiment dispatch_mode must be 'sparse' or 'dense', got 'none'"
        )

    if spec.dispatch_mode not in ("sparse", "dense", "none"):
        errors.append(
            f"[{spec.name}] Invalid dispatch_mode '{spec.dispatch_mode}', "
            f"must be one of: sparse, dense, none"
        )

    return errors


def validate_suite(specs: list[ExperimentSpec], config_root: Path) -> None:
    """Validate all selected specs upfront. Fail fast with a clear report."""
    all_errors: list[str] = []
    for spec in specs:
        all_errors.extend(validate_spec(spec, config_root))

    if all_errors:
        print(f"\n{'='*60}")
        print("  SUITE VALIDATION FAILED")
        print(f"{'='*60}")
        for err in all_errors:
            print(f"  X {err}")
        print(f"{'='*60}\n")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Override builder
# ---------------------------------------------------------------------------


def build_overrides(
    spec: ExperimentSpec,
    run_prefix: str,
    checkpoint_root: str,
    common_overrides: list[str],
    skip_prepare: bool,
) -> list[str]:
    """Build the exact Hydra override list for a spec. No implicit defaults."""
    overrides = [
        f"experiment={spec.experiment}",
        f"model={spec.model}",
        f"run_name={run_prefix}_{spec.name}",
        f"trainer.save_dir={Path(checkpoint_root) / f'{run_prefix}_{spec.name}'}",
    ]
    if spec.router is not None:
        overrides.append(f"router={spec.router}")
    overrides.append(f"trainer.prepare_data={'false' if not skip_prepare else 'true'}")
    overrides.extend(common_overrides)
    return overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories once and run the full dense/top1/top2/reflected suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )
    parser.add_argument(
        "--only",
        choices=list(SUITE_BY_NAME.keys()),
        action="append",
        default=[],
        help="Run only selected experiment(s). Can be provided multiple times.",
    )
    parser.add_argument("--run-prefix", default="tinystories")
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/checkpoints/tinystories",
        help="Root directory under which each suite run gets its own checkpoint folder.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── Select experiments ──
    selected = [spec for spec in SUITE if not args.only or spec.name in args.only]

    # ── Build common overrides ──
    common_overrides = ["data=tinystories", "trainer=tinystories", *args.set]
    if args.epochs is not None:
        common_overrides.append(f"trainer.max_epochs={args.epochs}")
    validate_override_syntax(common_overrides)

    # ── Print execution plan ──
    print(f"\n{'='*70}")
    print(f"  TinyStories Suite - {len(selected)} experiment(s)")
    print(f"{'='*70}")
    for spec in selected:
        mode_label = {
            "dense": "DENSE (soft)",
            "sparse": "SPARSE (top-k)",
            "none": "FFN (no MoE)",
        }[spec.dispatch_mode]
        router_info = spec.router or "(none)"
        print(f"  {spec.name:>10}  |  model={spec.model:<12}  router={router_info:<12}  mode={mode_label}")
    print(f"{'='*70}")
    print()

    # ── Validate everything upfront ──
    validate_suite(selected, CONFIG_ROOT)
    print("  [OK] All config files found and compatibility checks passed.\n")

    # ── Prepare data ──
    if not args.skip_prepare:
        prepare_overrides = ["data=tinystories"]
        if args.rebuild_data:
            prepare_overrides.append("data.rebuild_cache=true")
        validate_override_syntax(prepare_overrides)
        code = run_command(module_cmd("moe_route.cli.prepare_data", prepare_overrides), args.dry_run)
        if code != 0:
            raise SystemExit(code)

    # ── Execute experiments ──
    for i, spec in enumerate(selected, 1):
        print(f"\n[suite] ({i}/{len(selected)}) Starting: {spec.name} - {spec.description}")
        overrides = build_overrides(
            spec,
            run_prefix=args.run_prefix,
            checkpoint_root=args.checkpoint_root,
            common_overrides=common_overrides,
            skip_prepare=args.skip_prepare,
        )
        validate_override_syntax(overrides)
        if args.nproc_per_node > 1:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={args.nproc_per_node}",
                "-m",
                "moe_route.cli.train",
                "distributed=ddp",
                *overrides,
            ]
        else:
            cmd = module_cmd("moe_route.cli.train", overrides)
        code = run_command(cmd, args.dry_run)
        if code != 0:
            print(f"\n[suite] ERROR: Experiment '{spec.name}' failed with exit code {code}.")
            raise SystemExit(code)
        print(f"[suite] OK: Experiment '{spec.name}' completed successfully.")

    print(f"\n[suite] All {len(selected)} experiment(s) completed successfully.\n")


def _build_epilog() -> str:
    lines = ["Available experiments:\n"]
    for spec in SUITE:
        mode = {"dense": "dense", "sparse": "sparse", "none": "ffn"}[spec.dispatch_mode]
        router = spec.router or "-"
        lines.append(f"  {spec.name:<12}  {spec.description:<55}  [router={router}, mode={mode}]")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
