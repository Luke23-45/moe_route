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
        self.experts = nn.ModuleList(
            [ExpertMLP(d_model, expert_hidden_size, dropout) for _ in range(num_experts)]
        )
        self.last_diagnostics: RoutingDiagnostics | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        route = self.router(flat)

        num_tokens, top_k = route.expert_indices.shape
        flat_indices = route.expert_indices.view(-1)
        flat_mask = route.dispatch_mask.view(-1)
        token_ranks = route.token_ranks.view(-1)

        cap = route.diagnostics.capacity[0].item()
        num_experts = len(self.experts)

        dummy_idx = num_experts * (cap + 1) - 1
        safe_token_ranks = token_ranks.clamp(max=cap)
        flat_buffer_idx = torch.where(
            flat_mask,
            flat_indices * (cap + 1) + safe_token_ranks,
            torch.full_like(flat_indices, dummy_idx)
        )

        flat_buffer = torch.zeros(num_experts * (cap + 1), flat.shape[-1], dtype=flat.dtype, device=flat.device)
        flat_buffer.index_add_(0, flat_buffer_idx, flat.repeat_interleave(top_k, dim=0))

        buffer = flat_buffer.view(num_experts, cap + 1, flat.shape[-1])
        buffer_out = torch.zeros_like(buffer)

        for expert_id, expert in enumerate(self.experts):
            buffer_out[expert_id, :cap] = expert(buffer[expert_id, :cap])

        flat_buffer_out = buffer_out.view(num_experts * (cap + 1), flat.shape[-1])
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

