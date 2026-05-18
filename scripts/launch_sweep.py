from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform Hydra sweep launcher.")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cmd = [sys.executable, "-m", "moe_route.cli.sweep", "--multirun", *args.overrides]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

