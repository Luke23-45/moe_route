from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def collect_env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "git_commit": git_commit(),
    }


def write_run_metadata(output_dir: str | Path, cfg: Any) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (path / "environment.json").write_text(json.dumps(collect_env(), indent=2), encoding="utf-8")
