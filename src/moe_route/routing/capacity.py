from __future__ import annotations

from dataclasses import dataclass

import torch
from moe_route.routing.kernels import HAS_TRITON, enforce_capacity_triton


@dataclass(frozen=True)
class CapacityPolicy:
    num_experts: int
    top_k: int
    capacity_factor: float = 1.25
    drop_tokens: bool = True

    def capacity(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (self.num_experts,),
            self.capacity_per_expert(num_tokens),
            dtype=torch.long,
            device=device,
        )

    def capacity_per_expert(self, num_tokens: int) -> int:
        cap = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        return max(cap, 1)

    def enforce(
        self, expert_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid dispatch mask, accepted load, raw selected load, overflow counts, and token ranks."""
        flat = expert_indices.reshape(-1)
        device = flat.device
        capacity = self.capacity(expert_indices.shape[0], device)

        if HAS_TRITON and device.type == 'cuda':
            valid_mask, accepted, raw_load, overflow, token_ranks = enforce_capacity_triton(
                expert_indices, capacity, self.drop_tokens
            )
            # enforce_capacity_triton reshapes expert_indices to 1-D internally,
            # so valid_mask and token_ranks come back flat [T*K].
            # Reshape them to the caller's [T, K] shape to match the CPU path.
            return (
                valid_mask.reshape_as(expert_indices),
                accepted,
                raw_load,
                overflow,
                token_ranks.reshape_as(expert_indices),
            )

        raw_load = torch.bincount(flat, minlength=self.num_experts)

        # Standard Switch/GShard-style expert positions are cumulative ranks in
        # token order within each expert. Overflow assignments are then masked.
        order = torch.argsort(flat, stable=True)
        sorted_flat = flat.index_select(0, order)
        positions = torch.arange(sorted_flat.numel(), device=device, dtype=torch.long)
        is_group_start = torch.ones_like(sorted_flat, dtype=torch.bool)
        is_group_start[1:] = sorted_flat[1:] != sorted_flat[:-1]
        group_starts = torch.where(is_group_start, positions, torch.zeros_like(positions))
        group_starts = torch.cummax(group_starts, dim=0).values
        sorted_ranks = positions - group_starts
        token_ranks = torch.empty_like(sorted_ranks)
        token_ranks.scatter_(0, order, sorted_ranks)

        if self.drop_tokens:
            token_capacity = capacity.gather(0, flat)
            valid = token_ranks < token_capacity
        else:
            valid = torch.ones_like(flat, dtype=torch.bool)

        accepted = torch.bincount(flat[valid], minlength=self.num_experts)
        overflow = (raw_load - capacity).clamp_min(0)

        return valid.reshape_as(expert_indices), accepted, raw_load, overflow, token_ranks.reshape_as(expert_indices)
