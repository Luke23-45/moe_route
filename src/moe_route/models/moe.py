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
        output = torch.zeros_like(flat)

        for expert_id, expert in enumerate(self.experts):
            for slot in range(route.expert_indices.shape[1]):
                mask = (route.expert_indices[:, slot] == expert_id) & route.dispatch_mask[:, slot]
                expert_out = expert(flat[mask])
                output[mask] += expert_out * route.combine_weights[mask, slot].unsqueeze(-1)

        self.last_diagnostics = route.diagnostics
        return output.reshape(original_shape), route.diagnostics.aux_loss

    def pressure_state_dict(self) -> list[dict[str, torch.Tensor] | None]:
        return [self.router.pressure_state_dict()]

    def load_pressure_state_dict(self, states: list[dict[str, torch.Tensor] | None]) -> None:
        if states:
            self.router.load_pressure_state_dict(states[0])

