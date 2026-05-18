from __future__ import annotations

import json
from pathlib import Path


def summarize_routing_run(run_dir: str | Path) -> dict[str, float]:
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    router_keys = [key for row in records for key in row if key.startswith("router/")]
    summary: dict[str, float] = {}
    for key in sorted(set(router_keys)):
        values = [float(row[key]) for row in records if key in row]
        if values:
            summary[f"{key}/mean"] = sum(values) / len(values)
            summary[f"{key}/last"] = values[-1]
    return summary

