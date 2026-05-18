from __future__ import annotations

import argparse

from _runner import (
    add_common_config_args,
    module_cmd,
    overrides_from_args,
    run_command,
    validate_override_syntax,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform evaluation launcher.")
    add_common_config_args(parser)
    parser.add_argument("checkpoint")
    parser.add_argument("--tasks", action="store_true", help="Run downstream task adapter instead of PPL.")
    args = parser.parse_args()

    overrides = [f"checkpoint={args.checkpoint}", *overrides_from_args(args)]
    validate_override_syntax(overrides)
    module = "moe_route.cli.eval_tasks" if args.tasks else "moe_route.cli.eval_ppl"
    raise SystemExit(run_command(module_cmd(module, overrides), args.dry_run))


if __name__ == "__main__":
    main()

