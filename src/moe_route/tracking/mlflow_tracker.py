from __future__ import annotations

from pathlib import Path
from typing import Any


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str, run_name: str) -> None:
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: str(v) for k, v in params.items()}
        self.mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        self.mlflow.log_artifact(str(path))

    def close(self) -> None:
        self.mlflow.end_run()

