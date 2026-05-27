"""Per-run experiment manifest capture.

Generates a complete, self-describing manifest JSON for every experiment run.
The manifest records all information needed to reproduce or audit a run:
  - Full resolved Hydra configuration
  - Random seed
  - Git commit SHA
  - Environment metadata (Python version, PyTorch version, GPU info)
  - Wall-clock timing (start, end, duration)
  - Token throughput
  - Checkpoint paths
  - Evaluation output paths

Manifests are written both at run *start* (partial) and run *end* (complete)
so that interrupted runs are still auditable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from studies.utils.experiment_spec import ExperimentSpec


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or None if unavailable."""
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return None


def _collect_env() -> dict[str, Any]:
    """Collect environment metadata for the manifest."""
    import platform

    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device_count"] = torch.cuda.device_count()
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            env["cuda_compute_capability"] = f"{cap[0]}.{cap[1]}"
            env["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 2
            )
    except ImportError:
        pass

    return env


@dataclass
class RunManifest:
    """Complete metadata record for a single experiment run.

    Written as JSON to the persistence directory.  Fields are populated
    progressively: ``create()`` captures pre-run information, and
    ``finalize()`` adds post-run timing and results.
    """

    # --- Identity ---
    experiment_id: str = ""
    run_name: str = ""
    seed: int = 0
    study_name: str = ""

    # --- Spec metadata ---
    method: str = ""
    router: str | None = None
    model: str = ""
    data: str = ""
    trainer: str = ""
    dispatch_mode: str = ""
    hydra_overrides: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    # --- Provenance ---
    git_sha: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    # --- Resolved config ---
    resolved_config: dict[str, Any] | None = None

    # --- Timing ---
    start_time_iso: str = ""
    end_time_iso: str = ""
    wall_clock_seconds: float = 0.0

    # --- Results ---
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    checkpoint_path: str | None = None

    # --- Evaluation ---
    eval_outputs: dict[str, str] = field(default_factory=dict)

    # --- Status ---
    status: str = "pending"  # pending → running → completed / failed

    @classmethod
    def create(
        cls,
        spec: ExperimentSpec,
        seed: int,
        run_name: str,
        study_name: str,
    ) -> RunManifest:
        """Create a manifest at run start with pre-run information."""
        return cls(
            experiment_id=spec.experiment_id,
            run_name=run_name,
            seed=seed,
            study_name=study_name,
            method=spec.name,
            router=spec.router,
            model=spec.model,
            data=spec.data,
            trainer=spec.trainer,
            dispatch_mode=spec.dispatch_mode.value,
            hydra_overrides=list(spec.hydra_overrides),
            tags=dict(spec.tags),
            git_sha=_git_sha(),
            environment=_collect_env(),
            start_time_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            status="running",
        )

    def finalize(
        self,
        *,
        success: bool,
        checkpoint_path: str | None = None,
        total_tokens: int = 0,
        tokens_per_second: float = 0.0,
        final_train_loss: float | None = None,
        final_val_loss: float | None = None,
    ) -> None:
        """Update the manifest with post-run results."""
        self.end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if self.start_time_iso:
            try:
                from datetime import datetime

                start = datetime.fromisoformat(self.start_time_iso)
                end = datetime.fromisoformat(self.end_time_iso)
                self.wall_clock_seconds = (end - start).total_seconds()
            except Exception:
                pass
        self.status = "completed" if success else "failed"
        self.checkpoint_path = checkpoint_path
        self.total_tokens = total_tokens
        self.tokens_per_second = tokens_per_second
        self.final_train_loss = final_train_loss
        self.final_val_loss = final_val_loss

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write manifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Load a manifest from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
