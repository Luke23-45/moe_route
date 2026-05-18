from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
SRC_ROOT = REPO_ROOT / "src"


def config_choices(group: str) -> list[str]:
    group_dir = CONFIG_ROOT / group
    if not group_dir.exists():
        return []
    return sorted(path.stem for path in group_dir.glob("*.yaml"))


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    return env


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", choices=config_choices("experiment"))
    parser.add_argument("--data", choices=config_choices("data"))
    parser.add_argument("--router", choices=config_choices("router"))
    parser.add_argument("--model", choices=config_choices("model"))
    parser.add_argument("--tokenizer", choices=config_choices("tokenizer"))
    parser.add_argument("--trainer", choices=config_choices("trainer"))
    parser.add_argument("--tracking", choices=config_choices("tracking"))
    parser.add_argument("--run-name")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional Hydra override. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing.")


def overrides_from_args(args: argparse.Namespace) -> list[str]:
    overrides: list[str] = []
    for key in ("experiment", "data", "router", "model", "tokenizer", "trainer", "tracking"):
        value = getattr(args, key, None)
        if value:
            overrides.append(f"{key}={value}")
    if getattr(args, "run_name", None):
        overrides.append(f"run_name={args.run_name}")
    overrides.extend(args.set)
    return overrides


def run_command(cmd: list[str], dry_run: bool = False) -> int:
    rendered = " ".join(cmd)
    print(f"[runner] cwd={REPO_ROOT}")
    print(f"[runner] command={rendered}")
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO_ROOT, env=python_env())


def module_cmd(module: str, overrides: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *overrides]


def validate_override_syntax(overrides: list[str]) -> None:
    invalid = [item for item in overrides if "=" not in item and not item.startswith("+")]
    if invalid:
        raise SystemExit(f"Invalid Hydra overrides: {invalid}. Use KEY=VALUE syntax.")

