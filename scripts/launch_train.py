from __future__ import annotations

import argparse
import sys

from _runner import (
    add_common_config_args,
    module_cmd,
    overrides_from_args,
    run_command,
    validate_override_syntax,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intelligent cross-platform training launcher for MoE routing experiments."
    )
    add_common_config_args(parser)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument(
        "--skip-data-prepare",
        action="store_true",
        help="Require prepared data to already exist instead of preparing it before training.",
    )
    args = parser.parse_args()

    overrides = overrides_from_args(args)
    overrides.append(f"trainer.prepare_data={str(not args.skip_data_prepare).lower()}")
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
    raise SystemExit(run_command(cmd, args.dry_run))


if __name__ == "__main__":
    main()

