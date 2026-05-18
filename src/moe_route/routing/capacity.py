from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CapacityPolicy:
    num_experts: int
    top_k: int
    capacity_factor: float = 1.25
    drop_tokens: bool = True

    def capacity(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        cap = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        cap = max(cap, 1)
        return torch.full((self.num_experts,), cap, dtype=torch.long, device=device)

    def enforce(
        self, expert_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid dispatch mask, accepted load, raw selected load, and overflow counts."""
        flat = expert_indices.reshape(-1)
        device = flat.device
        capacity = self.capacity(expert_indices.shape[0], device)
        raw_load = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        valid = torch.ones_like(flat, dtype=torch.bool)

        raw_load.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        if self.drop_tokens:
            for expert in range(self.num_experts):
                positions = torch.nonzero(flat == expert, as_tuple=False).flatten()
                overflow = positions[capacity[expert] :]
                valid[overflow] = False

        accepted = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        if valid.any():
            accepted.scatter_add_(0, flat[valid], torch.ones_like(flat[valid], dtype=torch.long))
        overflow = (raw_load - capacity).clamp_min(0)
        return valid.reshape_as(expert_indices), accepted, raw_load, overflow
