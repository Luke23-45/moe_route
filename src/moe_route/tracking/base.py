from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ExperimentTracker(Protocol):
    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: str | Path) -> None: ...

    def close(self) -> None: ...

