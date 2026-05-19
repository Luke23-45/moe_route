from __future__ import annotations

from abc import ABC, abstractmethod
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
    z_loss_weight: float = 0.001
    pressure_lr: float = 0.05
    pressure_alpha: float = 1.0
    pressure_beta: float = 1.0
    pressure_gamma: float = 0.0
    pressure_decay: float = 0.0
    # Reflected controller (v2) specific fields
    temperature: float = 1.0
    pressure_scale: float = 1.0
    pressure_eps: float = 1e-8
    learnable_bias: bool = True


class Router(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> RoutingResult:
        """Route flattened token representations to experts."""

    def pressure_state_dict(self) -> dict[str, torch.Tensor] | None:
        return None

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        _ = state
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

        dispatch_mask, load, raw_load, overflow, token_ranks = self.capacity.enforce(indices)
        weights = weights * dispatch_mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(denom > 0, weights / denom.clamp_min(1e-8), weights)

        load_fraction = load.float() / load.sum().clamp_min(1)
        raw_load_fraction = raw_load.float() / raw_load.sum().clamp_min(1)
        mean_probs = probs.mean(dim=0)
        
        # SOTA Z-Loss to penalize large logits
        z_loss = self.cfg.z_loss_weight * (torch.logsumexp(logits, dim=-1) ** 2).mean()
        
        aux_loss = self.cfg.aux_loss_weight * self.cfg.num_experts * (
            mean_probs * raw_load_fraction
        ).sum() + z_loss
        
        capacity = self.capacity.capacity(x.shape[0], x.device)
        accepted_assignments = load.sum().float()
        requested_assignments = raw_load.sum().float().clamp_min(1.0)
        diagnostics = RoutingDiagnostics(
            raw_load=raw_load,
            load=load,
            load_fraction=load_fraction,
            raw_load_fraction=raw_load_fraction,
            capacity=capacity,
            dropped=(~dispatch_mask).all(dim=-1).float().mean(),
            dropped_assignments=(~dispatch_mask).float().mean(),
            entropy=routing_entropy(probs),
            overflow=overflow,
            aux_loss=aux_loss,
            capacity_utilization=load.float().sum() / capacity.float().sum().clamp_min(1.0),
            matched_compute_fraction=accepted_assignments / requested_assignments,
            z_loss=z_loss,
        )
        return RoutingResult(indices, weights, dispatch_mask, token_ranks, diagnostics)


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
        self._update_stream = None

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        self.pressure.to(x.device)
        return self.gate(x) - self.pressure.penalty().to(x.device)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        result = super().forward(x)
        if self.training:
            if x.device.type == "cuda":
                if self._update_stream is None:
                    self._update_stream = torch.cuda.Stream(device=x.device)
                with torch.cuda.stream(self._update_stream):
                    self.pressure.update(result.diagnostics.raw_load_fraction)
            else:
                self.pressure.update(result.diagnostics.raw_load_fraction)
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
    if cfg.kind == "reflected_v2":
        from moe_route.routing.reflected_controller import (
            ReflectedController,
            ReflectedControllerConfig,
        )

        return ReflectedController(
            ReflectedControllerConfig(
                d_model=cfg.d_model,
                num_experts=cfg.num_experts,
                temperature=cfg.temperature,
                pressure_scale=cfg.pressure_scale,
                pressure_lr=cfg.pressure_lr,
                pressure_beta=cfg.pressure_beta,
                pressure_eps=cfg.pressure_eps,
                pressure_decay=cfg.pressure_decay,
                learnable_bias=cfg.learnable_bias,
                z_loss_weight=cfg.z_loss_weight,
            )
        )
    raise ValueError(f"Unknown router kind: {cfg.kind}")
