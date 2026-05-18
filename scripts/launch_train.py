from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform training launcher.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=smoke_reflected")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    args = parser.parse_args()

    if args.nproc_per_node > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={args.nproc_per_node}",
            "-m",
            "moe_route.cli.train",
            "distributed=ddp",
            *args.overrides,
        ]
    else:
        cmd = [sys.executable, "-m", "moe_route.cli.train", *args.overrides]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

