from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlTracker:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.params_path = self.run_dir / "params.json"

    def log_params(self, params: dict[str, Any]) -> None:
        self.params_path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        record = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def log_artifact(self, path: str | Path) -> None:
        artifacts = self.run_dir / "artifacts.txt"
        with artifacts.open("a", encoding="utf-8") as f:
            f.write(str(path) + "\n")

    def close(self) -> None:
        return None

