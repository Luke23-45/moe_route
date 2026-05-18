from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RoutingDiagnostics:
    load: torch.Tensor
    load_fraction: torch.Tensor
    capacity: torch.Tensor
    dropped: torch.Tensor
    entropy: torch.Tensor
    overflow: torch.Tensor
    aux_loss: torch.Tensor
    pressure: torch.Tensor | None = None
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class RoutingResult:
    expert_indices: torch.Tensor
    combine_weights: torch.Tensor
    dispatch_mask: torch.Tensor
    diagnostics: RoutingDiagnostics

