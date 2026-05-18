from __future__ import annotations

from pathlib import Path

from moe_route.tracking.base import ExperimentTracker
from moe_route.tracking.jsonl import JsonlTracker
from moe_route.tracking.mlflow_tracker import MLflowTracker


def build_tracker(cfg, run_dir: str | Path) -> ExperimentTracker:
    if bool(cfg.tracking.mlflow.enabled):
        return MLflowTracker(
            tracking_uri=str(cfg.tracking.mlflow.tracking_uri),
            experiment_name=str(cfg.tracking.mlflow.experiment_name),
            run_name=str(cfg.run_name),
        )
    return JsonlTracker(run_dir)

