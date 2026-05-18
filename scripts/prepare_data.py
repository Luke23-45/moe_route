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
    parser = argparse.ArgumentParser(
        description="Prepare/download/tokenize/pack the configured dataset before training."
    )
    add_common_config_args(parser)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the prepared token shard even if the manifest matches.",
    )
    args = parser.parse_args()
    overrides = overrides_from_args(args)
    if args.rebuild:
        overrides.append("data.rebuild_cache=true")
    validate_override_syntax(overrides)
    raise SystemExit(run_command(module_cmd("moe_route.cli.prepare_data", overrides), args.dry_run))


if __name__ == "__main__":
    main()

