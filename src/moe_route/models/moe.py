from __future__ import annotations

import torch
from torch import nn

from moe_route.routing.routers import Router, RouterConfig, build_router
from moe_route.routing.types import RoutingDiagnostics


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BatchedExpertMLP(nn.Module):
    def __init__(self, num_experts: int, d_model: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden_size))
        self.b1 = nn.Parameter(torch.empty(num_experts, 1, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, d_model))
        self.b2 = nn.Parameter(torch.empty(num_experts, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        import math
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.w1[i], a=math.sqrt(5))
            fan_in_1, _ = nn.init._calculate_fan_in_and_fan_out(self.w1[i])
            bound_1 = 1 / math.sqrt(fan_in_1) if fan_in_1 > 0 else 0
            nn.init.uniform_(self.b1[i], -bound_1, bound_1)

            nn.init.kaiming_uniform_(self.w2[i], a=math.sqrt(5))
            fan_in_2, _ = nn.init._calculate_fan_in_and_fan_out(self.w2[i])
            bound_2 = 1 / math.sqrt(fan_in_2) if fan_in_2 > 0 else 0
            nn.init.uniform_(self.b2[i], -bound_2, bound_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.bmm(x, self.w1) + self.b1
        h = torch.nn.functional.gelu(h)
        h = self.dropout(h)
        return torch.bmm(h, self.w2) + self.b2


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_hidden_size: int,
        dropout: float,
        router_cfg: RouterConfig,
    ) -> None:
        super().__init__()
        self.router: Router = build_router(router_cfg)
        self.experts = BatchedExpertMLP(num_experts, d_model, expert_hidden_size, dropout)
        self.last_diagnostics: RoutingDiagnostics | None = None
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        route = self.router(flat)

        num_tokens, top_k = route.expert_indices.shape
        flat_indices = route.expert_indices.view(-1)
        flat_mask = route.dispatch_mask.view(-1)
        token_ranks = route.token_ranks.view(-1)

        cap = max(int(self.router.cfg.capacity_factor * num_tokens * self.router.cfg.top_k / self.router.cfg.num_experts), 1)

        dummy_idx = self.num_experts * (cap + 1) - 1
        safe_token_ranks = token_ranks.clamp(max=cap)
        flat_buffer_idx = torch.where(
            flat_mask,
            flat_indices * (cap + 1) + safe_token_ranks,
            torch.full_like(flat_indices, dummy_idx)
        )

        flat_buffer = torch.zeros(self.num_experts * (cap + 1), flat.shape[-1], dtype=flat.dtype, device=flat.device)
        flat_buffer.index_add_(0, flat_buffer_idx, flat.repeat_interleave(top_k, dim=0))

        buffer = flat_buffer.view(self.num_experts, cap + 1, flat.shape[-1])
        
        # SOTA Batched Matrix Multiplication (BMM) - Zero Python Overhead!
        buffer_out = torch.zeros_like(buffer)
        buffer_out[:, :cap, :] = self.experts(buffer[:, :cap, :])

        flat_buffer_out = buffer_out.view(self.num_experts * (cap + 1), flat.shape[-1])
        expert_out = flat_buffer_out.index_select(0, flat_buffer_idx)

        weights = route.combine_weights.view(-1) * flat_mask.to(flat.dtype)
        expert_out = expert_out * weights.unsqueeze(-1)

        output = expert_out.view(num_tokens, top_k, flat.shape[-1]).sum(dim=1)

        self.last_diagnostics = route.diagnostics
        return output.reshape(original_shape), route.diagnostics.aux_loss

    def pressure_state_dict(self) -> list[dict[str, torch.Tensor] | None]:
        return [self.router.pressure_state_dict()]

    def load_pressure_state_dict(self, states: list[dict[str, torch.Tensor] | None]) -> None:
        if states:
            self.router.load_pressure_state_dict(states[0])

