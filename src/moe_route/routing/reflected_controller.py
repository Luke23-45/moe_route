"""Reflected Controller router for the proposed reflected MoE variants only."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from moe_route.routing.metrics import routing_entropy
from moe_route.routing.routers import Router
from moe_route.routing.types import RoutingDiagnostics, RoutingResult
from moe_route.routing.kernels import HAS_TRITON, enforce_capacity_triton


@dataclass(frozen=True)
class ReflectedControllerConfig:
    """Configuration for the reflected router."""

    d_model: int
    num_experts: int
    temperature: float = 1.0
    pressure_scale: float = 1.0
    pressure_lr: float = 0.05
    pressure_beta: float = 1.0
    pressure_eps: float = 1e-8
    pressure_decay: float = 0.0
    pressure_warmup_steps: int = 1000
    learnable_bias: bool = True
    z_loss_weight: float = 0.0
    shared_experts: int = 0
    routing_mode: str = "dense"  # "dense" | "sparse"
    top_k: int | None = None
    capacity_factor: float = 1.25
    drop_tokens: bool = True
    gate_function: str = "softmax"  # "softmax" | "sigmoid"


class ReflectedPressureState(nn.Module):
    """Nonnegative expert-pressure state with projected dual ascent."""

    def __init__(self, cfg: ReflectedControllerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "capacity_weights",
            torch.full((cfg.num_experts,), 1.0 / cfg.num_experts, dtype=torch.float32),
        )
        self.register_buffer("q", torch.zeros(cfg.num_experts, dtype=torch.float32))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.base_lr = cfg.pressure_lr / (cfg.d_model ** 0.5)

    def penalty(self) -> torch.Tensor:
        """Compute the absorbed reflected penalty gamma * q."""
        return self.cfg.pressure_scale * self.q

    def current_lr(self) -> float:
        step = max(int(self.step.item()), 1)
        warmup = max(int(self.cfg.pressure_warmup_steps), 1)
        phase_scale = min(1.0, math.sqrt(warmup / step))
        return self.base_lr * phase_scale

    @torch.no_grad()
    def update_from_mass(self, mass: torch.Tensor, num_tokens: int) -> None:
        """Projected reflected ascent from accepted continuous routed mass."""
        self.step.add_(1)
        m = mass.detach().to(self.q.device, dtype=self.q.dtype) / num_tokens
        delta = m - self.capacity_weights

        if self.cfg.pressure_decay > 0:
            self.q.mul_(1.0 - self.cfg.pressure_decay)

        self.q.add_(self.current_lr() * delta)
        self.q.clamp_(min=0.0)

    def reset(self) -> None:
        self.q.zero_()


class ReflectedController(Router):
    """Reflected Controller MoE router with reflected-only updates."""

    def __init__(self, cfg: ReflectedControllerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        if cfg.routing_mode not in ("dense", "sparse"):
            raise ValueError(
                f"Invalid routing_mode '{cfg.routing_mode}'. Must be one of: 'dense', 'sparse'"
            )
        if cfg.gate_function not in ("softmax", "sigmoid"):
            raise ValueError(
                f"Invalid gate_function '{cfg.gate_function}'. Must be one of: 'softmax', 'sigmoid'"
            )
        if cfg.shared_experts < 0:
            raise ValueError("shared_experts must be non-negative")

        if cfg.routing_mode == "sparse":
            if cfg.top_k is None or cfg.top_k <= 0:
                raise ValueError(
                    "ReflectedController in 'sparse' mode requires a positive "
                    f"'top_k' parameter (got top_k={cfg.top_k})"
                )
            from moe_route.routing.capacity import CapacityPolicy

            self.capacity = CapacityPolicy(
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                capacity_factor=cfg.capacity_factor,
                drop_tokens=cfg.drop_tokens,
            )
        else:
            self.capacity = None

        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        if cfg.learnable_bias:
            self.bias = nn.Parameter(torch.zeros(cfg.num_experts))
        else:
            self.register_buffer("bias", torch.zeros(cfg.num_experts))
        self.pressure = ReflectedPressureState(cfg)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        num_tokens = x.shape[0]
        num_experts = self.cfg.num_experts
        device = x.device

        raw_affinity = self.gate(x)
        reflected_scores = raw_affinity + self.bias - self.pressure.penalty()
        full_probs = self._full_gate_probs(reflected_scores)

        if self.cfg.routing_mode == "sparse":
            result, feedback_mass = self._route_sparse(
                reflected_scores,
                full_probs,
                num_tokens,
                num_experts,
                device,
            )
        else:
            result, feedback_mass = self._route_dense(full_probs, num_tokens, num_experts, device)

        if self.training:
            self.pressure.update_from_mass(feedback_mass, num_tokens)

        z_loss = torch.zeros((), device=device)
        if self.cfg.z_loss_weight > 0:
            z_loss = self.cfg.z_loss_weight * (torch.logsumexp(raw_affinity, dim=-1) ** 2).mean()

        result.diagnostics.aux_loss = z_loss
        if self.cfg.z_loss_weight > 0:
            result.diagnostics.z_loss = z_loss
        result.diagnostics.pressure = self.pressure.q.detach().clone()
        return result

    def _full_gate_probs(self, reflected_scores: torch.Tensor) -> torch.Tensor:
        if self.cfg.gate_function == "sigmoid":
            p_raw = torch.sigmoid(reflected_scores / self.cfg.temperature)
            return p_raw / p_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.softmax(reflected_scores / self.cfg.temperature, dim=-1)

    def _route_dense(
        self,
        probs: torch.Tensor,
        num_tokens: int,
        num_experts: int,
        device: torch.device,
    ) -> tuple[RoutingResult, torch.Tensor]:
        feedback_mass = probs.detach().sum(dim=0)
        indices = torch.arange(num_experts, device=device).unsqueeze(0).expand(num_tokens, -1)
        dispatch_mask = torch.ones(num_tokens, num_experts, dtype=torch.bool, device=device)
        token_ranks = torch.zeros(num_tokens, num_experts, dtype=torch.long, device=device)

        load_fraction = feedback_mass / feedback_mass.sum().clamp_min(1e-8)
        diagnostics = RoutingDiagnostics(
            raw_load=feedback_mass,
            load=feedback_mass,
            load_fraction=load_fraction,
            raw_load_fraction=load_fraction,
            capacity=torch.full((num_experts,), float(num_tokens), device=device),
            dropped=torch.zeros((), device=device),
            dropped_assignments=torch.zeros((), device=device),
            entropy=routing_entropy(probs),
            overflow=torch.zeros(num_experts, device=device),
            aux_loss=torch.zeros((), device=device),
            capacity_utilization=torch.ones((), device=device),
            matched_compute_fraction=torch.ones((), device=device),
            pressure=self.pressure.q.detach().clone(),
            extra={"indices": indices, "dispatch_mask": dispatch_mask},
        )
        return RoutingResult(indices, probs, dispatch_mask, token_ranks, diagnostics), feedback_mass

    def _route_sparse(
        self,
        reflected_scores: torch.Tensor,
        full_probs: torch.Tensor,
        num_tokens: int,
        num_experts: int,
        device: torch.device,
    ) -> tuple[RoutingResult, torch.Tensor]:
        if self.cfg.gate_function == "sigmoid":
            sigmoid_scores = torch.sigmoid(reflected_scores / self.cfg.temperature)
            top_values, indices = torch.topk(sigmoid_scores, k=self.cfg.top_k, dim=-1)
            raw_combine_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            priorities = top_values
        else:
            top_scores, indices = torch.topk(reflected_scores, k=self.cfg.top_k, dim=-1)
            raw_combine_weights = torch.softmax(top_scores / self.cfg.temperature, dim=-1)
            priorities = top_scores

        dispatch_mask, load, raw_load, overflow, token_ranks = self._enforce_reflected_capacity(
            indices,
            priorities=priorities,
        )
        combine_weights = raw_combine_weights * dispatch_mask.to(raw_combine_weights.dtype)
        combine_denom = combine_weights.sum(dim=-1, keepdim=True)
        combine_weights = torch.where(
            combine_denom > 0,
            combine_weights / combine_denom.clamp_min(1e-8),
            combine_weights,
        )

        feedback_load = torch.zeros(num_experts, dtype=full_probs.dtype, device=device)
        feedback_load.scatter_add_(0, indices.reshape(-1), combine_weights.detach().reshape(-1))
        requested_feedback_mass = full_probs.detach().sum(dim=0)

        load_fraction = feedback_load / feedback_load.sum().clamp_min(1e-8)
        raw_load_fraction = requested_feedback_mass / requested_feedback_mass.sum().clamp_min(1e-8)
        capacity = self.capacity.capacity(num_tokens, device)
        accepted_assignments = load.sum().float()
        requested_assignments = raw_load.sum().float().clamp_min(1.0)

        diagnostics = RoutingDiagnostics(
            raw_load=requested_feedback_mass,
            load=feedback_load,
            load_fraction=load_fraction,
            raw_load_fraction=raw_load_fraction,
            capacity=capacity,
            dropped=(~dispatch_mask).all(dim=-1).float().mean(),
            dropped_assignments=(~dispatch_mask).float().mean(),
            entropy=routing_entropy(full_probs),
            overflow=overflow,
            aux_loss=torch.zeros((), device=device),
            capacity_utilization=accepted_assignments / capacity.sum().clamp_min(1.0),
            matched_compute_fraction=accepted_assignments / requested_assignments,
            pressure=self.pressure.q.detach().clone(),
            extra={
                "indices": indices,
                "dispatch_mask": dispatch_mask,
                "raw_assignment_load": raw_load,
                "accepted_assignment_load": load,
                "feedback_weights": combine_weights,
                "requested_feedback_mass": requested_feedback_mass,
            },
        )

        return (
            RoutingResult(indices, combine_weights, dispatch_mask, token_ranks, diagnostics),
            feedback_load,
        )

    def _enforce_reflected_capacity(
        self,
        expert_indices: torch.Tensor,
        priorities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reflected sparse capacity admission, isolated from baseline routers."""
        if priorities.shape != expert_indices.shape:
            raise ValueError(
                f"priorities shape {tuple(priorities.shape)} must match "
                f"expert_indices shape {tuple(expert_indices.shape)}"
            )

        flat = expert_indices.reshape(-1)
        flat_priorities = priorities.reshape(-1)
        device = flat.device
        capacity = self.capacity.capacity(expert_indices.shape[0], device)
        
        # Fast path: use fused Triton kernel if available (drops by arrival order, not priority)
        # SOTA implementations avoid sorting to maximize bandwidth
        if HAS_TRITON and device.type == 'cuda':
            return enforce_capacity_triton(expert_indices, capacity, self.cfg.drop_tokens)

        raw_load = torch.bincount(flat, minlength=self.cfg.num_experts)

        # CPU/Fallback path: Optimized single argsort using a composite key
        # We want to group by expert ID, and within each expert sort by descending priority.
        # Since priorities are typically bounded, we can normalize them to [0, 1).
        p_min = flat_priorities.min()
        p_max = flat_priorities.max()
        norm_priority = (flat_priorities - p_min) / (p_max - p_min + 1e-6)
        
        # Composite key: expert_id - norm_priority (so higher priority comes first within the same expert)
        composite_key = flat.float() - norm_priority
        order = torch.argsort(composite_key, stable=True)

        sorted_flat = flat.index_select(0, order)
        positions = torch.arange(sorted_flat.numel(), device=device, dtype=torch.long)
        is_group_start = torch.ones_like(sorted_flat, dtype=torch.bool)
        is_group_start[1:] = sorted_flat[1:] != sorted_flat[:-1]
        group_starts = torch.where(is_group_start, positions, torch.zeros_like(positions))
        group_starts = torch.cummax(group_starts, dim=0).values
        sorted_ranks = positions - group_starts
        token_ranks = torch.empty_like(sorted_ranks)
        token_ranks.scatter_(0, order, sorted_ranks)

        if self.cfg.drop_tokens:
            token_capacity = capacity.gather(0, flat)
            valid = token_ranks < token_capacity
        else:
            valid = torch.ones_like(flat, dtype=torch.bool)

        accepted = torch.bincount(flat[valid], minlength=self.cfg.num_experts)
        overflow = (raw_load - capacity).clamp_min(0)

        return (
            valid.reshape_as(expert_indices),
            accepted,
            raw_load,
            overflow,
            token_ranks.reshape_as(expert_indices),
        )

    def pressure_state_dict(self) -> dict[str, torch.Tensor]:
        return self.pressure.state_dict()

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        if state is not None:
            self.pressure.load_state_dict(state)
            self.pressure.q.clamp_(min=0.0)
