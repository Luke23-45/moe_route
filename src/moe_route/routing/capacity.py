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
        self, expert_indices: torch.Tensor,
        priorities: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid dispatch mask, accepted load, raw selected load, overflow counts, and token ranks."""
        flat = expert_indices.reshape(-1)
        device = flat.device
        capacity = self.capacity(expert_indices.shape[0], device)

        raw_load = torch.bincount(flat, minlength=self.num_experts)

        # Rank assignments within each expert. When routing priorities are supplied,
        # capacity is awarded to the strongest assignments instead of arbitrary token
        # order; this removes a sequence-order bottleneck in sparse dispatch.
        if priorities is not None:
            if priorities.shape != expert_indices.shape:
                raise ValueError(
                    f"priorities shape {tuple(priorities.shape)} must match "
                    f"expert_indices shape {tuple(expert_indices.shape)}"
                )
            flat_priorities = priorities.reshape(-1)
            priority_order = torch.argsort(-flat_priorities, stable=True)
            priority_sorted_experts = flat.index_select(0, priority_order)
            expert_order = torch.argsort(priority_sorted_experts, stable=True)
            order = priority_order.index_select(0, expert_order)
        else:
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
