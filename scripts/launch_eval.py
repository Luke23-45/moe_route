from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform evaluation launcher.")
    parser.add_argument("checkpoint")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--tasks", action="store_true", help="Run downstream task adapter instead of PPL.")
    args = parser.parse_args()
    module = "moe_route.cli.eval_tasks" if args.tasks else "moe_route.cli.eval_ppl"
    cmd = [sys.executable, "-m", module, f"checkpoint={args.checkpoint}", *args.overrides]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

