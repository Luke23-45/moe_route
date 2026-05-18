from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RoutingDiagnostics:
    raw_load: torch.Tensor
    load: torch.Tensor
    load_fraction: torch.Tensor
    raw_load_fraction: torch.Tensor
    capacity: torch.Tensor
    dropped: torch.Tensor
    dropped_assignments: torch.Tensor
    entropy: torch.Tensor
    overflow: torch.Tensor
    aux_loss: torch.Tensor
    capacity_utilization: torch.Tensor
    matched_compute_fraction: torch.Tensor
    pressure: torch.Tensor | None = None
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class RoutingResult:
    expert_indices: torch.Tensor
    combine_weights: torch.Tensor
    dispatch_mask: torch.Tensor
    token_ranks: torch.Tensor
    diagnostics: RoutingDiagnostics
