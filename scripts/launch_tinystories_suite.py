from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from _runner import module_cmd, run_command, validate_override_syntax


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    experiment: str
    model: str
    router: str | None = None


SUITE = (
    ExperimentSpec("dense", "tinystories_dense", "tiny_dense"),
    ExperimentSpec("top1", "tinystories_top1", "tiny_moe", "top1"),
    ExperimentSpec("top2", "tinystories_top2", "tiny_moe", "top2"),
    ExperimentSpec("reflected", "tinystories_reflected", "tiny_moe", "reflected_top2"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories once and run the full dense/top1/top2/reflected suite."
    )
    parser.add_argument(
        "--only",
        choices=[spec.name for spec in SUITE],
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

    selected = [spec for spec in SUITE if not args.only or spec.name in args.only]
    common_overrides = ["data=tinystories", "trainer=tinystories", *args.set]
    if args.epochs is not None:
        common_overrides.append(f"trainer.max_epochs={args.epochs}")
    validate_override_syntax(common_overrides)

    if not args.skip_prepare:
        prepare_overrides = ["data=tinystories"]
        if args.rebuild_data:
            prepare_overrides.append("data.rebuild_cache=true")
        validate_override_syntax(prepare_overrides)
        code = run_command(module_cmd("moe_route.cli.prepare_data", prepare_overrides), args.dry_run)
        if code != 0:
            raise SystemExit(code)

    for spec in selected:
        overrides = [
            f"experiment={spec.experiment}",
            f"model={spec.model}",
            f"run_name={args.run_prefix}_{spec.name}",
            f"trainer.save_dir={Path(args.checkpoint_root) / f'{args.run_prefix}_{spec.name}'}",
            "trainer.prepare_data=false" if not args.skip_prepare else "trainer.prepare_data=true",
            *common_overrides,
        ]
        if spec.router is not None:
            overrides.insert(1, f"router={spec.router}")
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
            raise SystemExit(code)


if __name__ == "__main__":
    main()
