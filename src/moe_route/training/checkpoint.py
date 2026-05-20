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


def migrate_state_dict_to_batched(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    expert_params = {}
    for key, tensor in state_dict.items():
        if ".ff.experts." in key and (".net." in key):
            parts = key.split(".ff.experts.")
            prefix = parts[0] + ".ff.experts"
            suffix = parts[1]
            expert_id = int(suffix.split(".")[0])
            param_path = ".".join(suffix.split(".")[1:])
            
            group_key = (prefix, param_path)
            if group_key not in expert_params:
                expert_params[group_key] = {}
            expert_params[group_key][expert_id] = tensor
        else:
            new_state_dict[key] = tensor
            
    for (prefix, param_path), expert_tensors in expert_params.items():
        num_experts = len(expert_tensors)
        if param_path == "net.0.weight":
            stacked = torch.stack([expert_tensors[i] for i in range(num_experts)])
            new_state_dict[f"{prefix}.w1"] = stacked.transpose(1, 2)
        elif param_path == "net.0.bias":
            stacked = torch.stack([expert_tensors[i] for i in range(num_experts)])
            new_state_dict[f"{prefix}.b1"] = stacked.unsqueeze(1)
        elif param_path == "net.3.weight":
            stacked = torch.stack([expert_tensors[i] for i in range(num_experts)])
            new_state_dict[f"{prefix}.w2"] = stacked.transpose(1, 2)
        elif param_path == "net.3.bias":
            stacked = torch.stack([expert_tensors[i] for i in range(num_experts)])
            new_state_dict[f"{prefix}.b2"] = stacked.unsqueeze(1)
            
    return new_state_dict


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: Optimizer | None = None, scheduler=None) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = unwrap_model(model)
    
    state_dict = payload["model"]
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    state_dict = migrate_state_dict_to_batched(state_dict)
    missing_keys, unexpected_keys = raw_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[checkpoint] WARNING: Missing keys in checkpoint (randomly initialized): {missing_keys}")
    if unexpected_keys:
        print(f"[checkpoint] WARNING: Unexpected keys in checkpoint (ignored): {unexpected_keys}")
    if payload.get("pressure") is not None and hasattr(raw_model, "load_pressure_state_dict"):
        raw_model.load_pressure_state_dict(payload["pressure"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng")
    if isinstance(rng, dict) and rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(payload.get("step", 0))
