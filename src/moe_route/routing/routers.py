from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moe_route.routing.capacity import CapacityPolicy
from moe_route.routing.metrics import routing_entropy
from moe_route.routing.pressure import PressureState, PressureStateConfig
from moe_route.routing.types import RoutingDiagnostics, RoutingResult


@dataclass(frozen=True)
class RouterConfig:
    kind: str
    d_model: int
    num_experts: int
    top_k: int = 1
    capacity_factor: float = 1.25
    drop_tokens: bool = True
    aux_loss_weight: float = 0.01
    pressure_lr: float = 0.05
    pressure_alpha: float = 1.0
    pressure_beta: float = 1.0
    pressure_gamma: float = 0.0
    pressure_decay: float = 0.0


class Router(nn.Module):
    def forward(self, x: torch.Tensor) -> RoutingResult:  # pragma: no cover - interface
        raise NotImplementedError

    def pressure_state_dict(self) -> dict[str, torch.Tensor] | None:
        return None

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        return None


class TopKRouter(Router):
    def __init__(self, cfg: RouterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.capacity = CapacityPolicy(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            capacity_factor=cfg.capacity_factor,
            drop_tokens=cfg.drop_tokens,
        )

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        logits = self.scores(x)
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        dispatch_mask, load, overflow = self.capacity.enforce(indices)
        weights = weights * dispatch_mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = torch.where(denom > 0, weights / denom, weights)

        load_fraction = load.float() / load.sum().clamp_min(1)
        mean_probs = probs.mean(dim=0)
        aux_loss = self.cfg.aux_loss_weight * self.cfg.num_experts * (mean_probs * load_fraction).sum()
        diagnostics = RoutingDiagnostics(
            load=load,
            load_fraction=load_fraction,
            capacity=self.capacity.capacity(x.shape[0], x.device),
            dropped=(~dispatch_mask).any(dim=-1).float().mean(),
            entropy=routing_entropy(probs),
            overflow=overflow,
            aux_loss=aux_loss,
        )
        return RoutingResult(indices, weights, dispatch_mask, diagnostics)


class ReflectedRouter(TopKRouter):
    def __init__(self, cfg: RouterConfig) -> None:
        super().__init__(cfg)
        self.pressure = PressureState(
            PressureStateConfig(
                num_experts=cfg.num_experts,
                lr=cfg.pressure_lr,
                alpha=cfg.pressure_alpha,
                beta=cfg.pressure_beta,
                gamma=cfg.pressure_gamma,
                decay=cfg.pressure_decay,
            )
        )

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        self.pressure.to(x.device)
        return self.gate(x) - self.pressure.penalty().to(x.device)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        result = super().forward(x)
        if self.training:
            self.pressure.update(result.diagnostics.load_fraction)
        result.diagnostics.pressure = self.pressure.q.detach().clone()
        return result

    def pressure_state_dict(self) -> dict[str, torch.Tensor]:
        return self.pressure.state_dict()

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        if state is not None:
            self.pressure.load_state_dict(state)


def build_router(cfg: RouterConfig) -> Router:
    if cfg.kind == "topk":
        return TopKRouter(cfg)
    if cfg.kind == "reflected":
        return ReflectedRouter(cfg)
    raise ValueError(f"Unknown router kind: {cfg.kind}")

