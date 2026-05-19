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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid dispatch mask, accepted load, raw selected load, overflow counts, and token ranks."""
        flat = expert_indices.reshape(-1)
        device = flat.device
        capacity = self.capacity(expert_indices.shape[0], device)
        
        raw_load = torch.bincount(flat, minlength=self.num_experts)
        
        # Compute token ranks using cumsum on one-hot representation
        one_hot = torch.nn.functional.one_hot(flat, num_classes=self.num_experts)
        expert_token_ranks = torch.cumsum(one_hot, dim=0) - 1
        token_ranks = expert_token_ranks.gather(1, flat.unsqueeze(1)).squeeze(1)

        if self.drop_tokens:
            token_capacity = capacity.gather(0, flat)
            valid = token_ranks < token_capacity
        else:
            valid = torch.ones_like(flat, dtype=torch.bool)

        accepted = torch.bincount(flat[valid], minlength=self.num_experts)
        overflow = (raw_load - capacity).clamp_min(0)
        
        return valid.reshape_as(expert_indices), accepted, raw_load, overflow, token_ranks.reshape_as(expert_indices)
