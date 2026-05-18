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
    parser = argparse.ArgumentParser(description="Cross-platform Hydra sweep launcher.")
    add_common_config_args(parser)
    args = parser.parse_args()
    overrides = ["--multirun", *overrides_from_args(args)]
    validate_override_syntax([item for item in overrides if item != "--multirun"])
    raise SystemExit(run_command(module_cmd("moe_route.cli.sweep", overrides), args.dry_run))


if __name__ == "__main__":
    main()

