from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    step: int,
    cfg: Any,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "pressure": raw_model.pressure_state_dict() if hasattr(raw_model, "pressure_state_dict") else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "config": cfg,
    }
    torch.save(payload, output)
    return output


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: Optimizer | None = None, scheduler=None) -> int:
    payload = torch.load(path, map_location="cpu")
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(payload["model"])
    if payload.get("pressure") is not None and hasattr(raw_model, "load_pressure_state_dict"):
        raw_model.load_pressure_state_dict(payload["pressure"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("step", 0))

