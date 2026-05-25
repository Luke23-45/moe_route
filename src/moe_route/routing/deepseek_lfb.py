"""DeepSeek Loss-Free Balancing router baseline.

This module intentionally does not share ReflectedController state or update
logic. DeepSeek-LFB uses an expert-wise routing bias updated outside the
gradient graph after optimizer steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from moe_route.routing.capacity import CapacityPolicy
from moe_route.routing.metrics import routing_entropy
from moe_route.routing.routers import Router
from moe_route.routing.types import RoutingDiagnostics, RoutingResult


@dataclass(frozen=True)
class DeepSeekLFBRouterConfig:
    d_model: int
    num_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25
    drop_tokens: bool = True
    z_loss_weight: float = 0.0
    bias_update_rate: float = 1e-3
    gate_function: str = "softmax"  # "softmax" | "sigmoid"


class DeepSeekLFBRouter(Router):
    """Auxiliary-loss-free Top-K router with post-step bias correction.

    The non-trainable bias affects expert selection only. Combine weights are
    computed from the original gate logits for the selected experts, matching
    the DeepSeek-LFB rule that the bias should not enter the gating weights.
    """

    def __init__(self, cfg: DeepSeekLFBRouterConfig) -> None:
        super().__init__()
        if cfg.top_k <= 0:
            raise ValueError(f"DeepSeekLFBRouter requires top_k > 0, got {cfg.top_k}")
        if cfg.top_k > cfg.num_experts:
            raise ValueError(
                f"DeepSeekLFBRouter top_k={cfg.top_k} exceeds num_experts={cfg.num_experts}"
            )
        if cfg.gate_function not in ("softmax", "sigmoid"):
            raise ValueError(
                f"Invalid gate_function '{cfg.gate_function}'. Must be one of: 'softmax', 'sigmoid'"
            )

        self.cfg = cfg
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        self.capacity = CapacityPolicy(
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
            capacity_factor=cfg.capacity_factor,
            drop_tokens=cfg.drop_tokens,
        )
        self.register_buffer("bias", torch.zeros(cfg.num_experts, dtype=torch.float32))
        self.register_buffer("_pending_raw_load", torch.zeros(cfg.num_experts, dtype=torch.float32))
        self.register_buffer("_has_pending_load", torch.zeros((), dtype=torch.bool))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        logits = self.scores(x)
        selection_scores = self._selection_scores(logits)
        _, indices = torch.topk(selection_scores, k=self.cfg.top_k, dim=-1)
        raw_selected_logits = logits.gather(dim=-1, index=indices)
        weights = self._combine_weights(raw_selected_logits)

        dispatch_mask, load, raw_load, overflow, token_ranks = self.capacity.enforce(indices)
        weights = weights * dispatch_mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(denom > 0, weights / denom.clamp_min(1e-8), weights)

        if self.training:
            self._pending_raw_load.add_(raw_load.detach().to(self._pending_raw_load))
            self._has_pending_load.fill_(True)

        raw_load_fraction = raw_load.float() / raw_load.sum().clamp_min(1)
        load_fraction = load.float() / load.sum().clamp_min(1)
        full_probs = self._full_gate_probs(logits)
        z_loss = torch.zeros((), device=x.device)
        if self.cfg.z_loss_weight > 0:
            z_loss = self.cfg.z_loss_weight * (torch.logsumexp(logits, dim=-1) ** 2).mean()

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
            entropy=routing_entropy(full_probs),
            overflow=overflow,
            aux_loss=z_loss,
            capacity_utilization=accepted_assignments / capacity.float().sum().clamp_min(1.0),
            matched_compute_fraction=accepted_assignments / requested_assignments,
            z_loss=z_loss if self.cfg.z_loss_weight > 0 else None,
            extra={
                "indices": indices,
                "dispatch_mask": dispatch_mask,
                "lfb_bias": self.bias.detach().clone(),
            },
        )
        return RoutingResult(indices, weights, dispatch_mask, token_ranks, diagnostics)

    def _selection_scores(self, logits: torch.Tensor) -> torch.Tensor:
        biased_logits = logits + self.bias.to(dtype=logits.dtype, device=logits.device)
        if self.cfg.gate_function == "sigmoid":
            return torch.sigmoid(biased_logits)
        return biased_logits

    def _combine_weights(self, selected_logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.gate_function == "sigmoid":
            values = torch.sigmoid(selected_logits)
            return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.softmax(selected_logits, dim=-1)

    def _full_gate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.cfg.gate_function == "sigmoid":
            values = torch.sigmoid(logits)
            return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def post_optimizer_step(self, *, distributed: bool = False) -> None:
        if not bool(self._has_pending_load.item()):
            return

        load = self._pending_raw_load.clone()
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)

        total = load.sum()
        if total > 0:
            load_fraction = load / total
            target = 1.0 / self.cfg.num_experts
            direction = torch.where(
                load_fraction > target,
                torch.full_like(load_fraction, -1.0),
                torch.full_like(load_fraction, 1.0),
            )
            self.bias.add_(self.cfg.bias_update_rate * direction.to(self.bias.dtype))

        self._pending_raw_load.zero_()
        self._has_pending_load.fill_(False)

    def pressure_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "bias": self.bias.detach().clone(),
            "pending_raw_load": self._pending_raw_load.detach().clone(),
            "has_pending_load": self._has_pending_load.detach().clone(),
        }

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        if state is None:
            return
        if "bias" in state:
            self.bias.copy_(state["bias"].to(device=self.bias.device, dtype=self.bias.dtype))
        if "pending_raw_load" in state:
            self._pending_raw_load.copy_(
                state["pending_raw_load"].to(
                    device=self._pending_raw_load.device,
                    dtype=self._pending_raw_load.dtype,
                )
            )
        if "has_pending_load" in state:
            self._has_pending_load.copy_(state["has_pending_load"].to(self._has_pending_load))
