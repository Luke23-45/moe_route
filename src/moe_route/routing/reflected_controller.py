"""Standalone Reflected Controller MoE Router.

Implements the reflected controller architecture from proposition_2.md:
  ReflectedMoE is an entropy-regularized, capacity-constrained MoE router
  whose expert-selection scores combine token affinity with a reflected
  nonnegative expert-pressure state, where the pressure is updated by
  projected dual ascent from recent routed mass and the core model is
  trained without auxiliary routing losses.

This module is fully independent of TopKRouter and CapacityPolicy.
No top-k selection, no capacity enforcement, no token dropping.
Pure dense soft routing with reflected pressure dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from moe_route.routing.metrics import routing_entropy
from moe_route.routing.routers import Router
from moe_route.routing.types import RoutingDiagnostics, RoutingResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReflectedControllerConfig:
    """Configuration for the Reflected Controller router.

    All field names map directly to the proposition_2.md notation:
        d_model      – input dimension
        num_experts  – E_ℓ
        temperature  – τ_ℓ  (softmax temperature)
        pressure_scale – ρ_ℓ  (router scale)
        pressure_lr  – η_ℓ  (pressure step size)
        pressure_beta – β   (capacity weight exponent in φ)
        pressure_eps – ε    (denominator floor)
        pressure_decay – optional EMA decay on pressure state
        learnable_bias – whether expert bias b_ℓ is learnable
        z_loss_weight – z-loss on raw gate logits only (0 = off, §4)
        gate_function – activation for routing: "softmax" (default) or
                        "sigmoid" (DeepSeek-V3 style independent gates)
    """

    d_model: int
    num_experts: int
    temperature: float = 1.0
    pressure_scale: float = 1.0
    pressure_lr: float = 0.05
    pressure_beta: float = 1.0
    pressure_eps: float = 1e-8
    pressure_decay: float = 0.0
    learnable_bias: bool = True
    z_loss_weight: float = 0.0
    # Mode selection and capacity parameters for sparse mode
    routing_mode: str = "dense"  # "dense" | "sparse"
    top_k: int | None = None
    capacity_factor: float = 1.25
    drop_tokens: bool = True
    gate_function: str = "softmax"  # "softmax" | "sigmoid"


# ---------------------------------------------------------------------------
# Reflected Pressure State (self-contained)
# ---------------------------------------------------------------------------

class ReflectedPressureState(nn.Module):
    """Nonnegative expert-pressure state with projected dual ascent.

    §2: q_e^(n+1) = max(0, q_e^(n) + η·(m_e - c_e))

    The pressure penalty used in routing scores is (§1):
        φ_e(q_e) = q_e / (μ_e^β + ε)

    where μ_e is the per-expert target capacity weight (uniform by default).
    """

    def __init__(self, cfg: ReflectedControllerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Uniform capacity weights: μ_e = 1/E  (§1 line 41-44)
        self.register_buffer(
            "capacity_weights",
            torch.full((cfg.num_experts,), 1.0 / cfg.num_experts, dtype=torch.float32)
        )
        # Pressure state: q ∈ ℝ₊^E, initialized to zero
        self.register_buffer("q", torch.zeros(cfg.num_experts, dtype=torch.float32))

    def penalty(self) -> torch.Tensor:
        """Compute ρ · φ(q) = ρ · q / (μ^β + ε).  (§1 line 56-63)"""
        denom = self.capacity_weights.clamp_min(self.cfg.pressure_eps).pow(self.cfg.pressure_beta)
        return self.cfg.pressure_scale * self.q / (denom + self.cfg.pressure_eps)

    @torch.no_grad()
    def update_from_mass(self, mass: torch.Tensor, num_tokens: int) -> None:
        """Projected reflected ascent from soft routed mass.

        §2 line 96-107:
            q_e ← max(0, q_e + η·(m_e − c_e))

        where m_e = Σ_t p̃_{t,e} is the soft routed mass and
        c_e = T / E is the uniform target capacity.
        """
        # Normalize by num_tokens so the delta is a fraction [-1, 1],
        # keeping the pressure update invariant to batch size.
        m = mass.detach().to(self.q.device, dtype=self.q.dtype) / num_tokens
        target = 1.0 / self.cfg.num_experts  # fractional target c_e
        delta = m - target

        if self.cfg.pressure_decay > 0:
            self.q.mul_(1.0 - self.cfg.pressure_decay)

        self.q.add_(self.cfg.pressure_lr * delta)
        # §2 line 110: Π_{[0,∞)}(z) = max{0, z}
        self.q.clamp_(min=0.0)

    def reset(self) -> None:
        self.q.zero_()


# ---------------------------------------------------------------------------
# Reflected Controller Router
# ---------------------------------------------------------------------------

class ReflectedController(Router):
    """Standalone Reflected Controller MoE Router.

    Computes dense soft routing over ALL experts with reflected pressure
    dynamics for load balancing. No top-k, no capacity enforcement,
    no token dropping, no auxiliary balance loss.

    Forward pass (proposition §1-§4):
        1. a = G(x)                              raw affinity
        2. s = a + b − ρ·φ(q)                    reflected score
        3. p = gate(s / τ)                        routing distribution
           where gate = softmax (default) or sigmoid (DeepSeek-V3)
        4. pressure update: q ← max(0, q + η(m−c)) projected ascent
        5. z-loss on a only (if enabled)          ablation, off by default

    Output is a RoutingResult with:
        expert_indices:  [T, E]  (all experts for every token)
        combine_weights: [T, E]  (soft probabilities)
        dispatch_mask:   [T, E]  (all True — no dropping)
    """

    def __init__(self, cfg: ReflectedControllerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── State Machine Configuration Validation ──
        if cfg.routing_mode not in ("dense", "sparse"):
            raise ValueError(
                f"Invalid routing_mode '{cfg.routing_mode}'. "
                f"Must be one of: 'dense', 'sparse'"
            )
        if cfg.gate_function not in ("softmax", "sigmoid"):
            raise ValueError(
                f"Invalid gate_function '{cfg.gate_function}'. "
                f"Must be one of: 'softmax', 'sigmoid'"
            )

        if cfg.routing_mode == "sparse":
            if cfg.top_k is None or cfg.top_k <= 0:
                raise ValueError(
                    f"ReflectedController in 'sparse' mode requires a positive 'top_k' "
                    f"parameter (got top_k={cfg.top_k})"
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

        # Gate network G_ℓ: ℝ^d → ℝ^E  (§1 line 28-30)
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        # Expert bias b_ℓ ∈ ℝ^E  (§1 line 32-34)
        if cfg.learnable_bias:
            self.bias = nn.Parameter(torch.zeros(cfg.num_experts))
        else:
            self.register_buffer("bias", torch.zeros(cfg.num_experts))

        # Reflected pressure state  (§1 line 36-38)
        self.pressure = ReflectedPressureState(cfg)

    def forward(self, x: torch.Tensor) -> RoutingResult:
        """Route tokens to experts with reflected scoring.

        Args:
            x: [T, d_model] flattened token representations.

        Returns:
            RoutingResult in either Dense (Mode A) or Sparse (Mode B) layout.
        """
        T = x.shape[0]
        E = self.cfg.num_experts
        device = x.device

        # ── §1 line 48-49: raw affinity a = G(x) ──
        a = self.gate(x)  # [T, E]

        # ── §1 line 53-57: reflected score s = a + b − ρ·φ(q) ──
        penalty = self.pressure.penalty()  # [E]
        s = a + self.bias - penalty  # [T, E]

        # ── Polymorphic routing separation ──
        if self.cfg.routing_mode == "sparse":
            result, m = self._route_sparse(s, T, E, device)
        else:
            result, m = self._route_dense(s, T, E, device)

        # ── §2 line 96-107: projected reflected ascent (training only) ──
        if self.training:
            self.pressure.update_from_mass(m, T)

        # ── §4 line 176-186: z-loss on RAW gate logits a, NOT pressured s ──
        z_loss = torch.zeros((), device=device)
        if self.cfg.z_loss_weight > 0:
            z_loss = self.cfg.z_loss_weight * (torch.logsumexp(a, dim=-1) ** 2).mean()

        # Update diagnostics aux_loss to hold the computed z_loss
        result.diagnostics.aux_loss = z_loss
        if self.cfg.z_loss_weight > 0:
            result.diagnostics.z_loss = z_loss

        # Ensure diagnostics contain the updated pressure
        result.diagnostics.pressure = self.pressure.q.detach().clone()

        return result

    def _route_dense(
        self, s: torch.Tensor, T: int, E: int, device: torch.device
    ) -> tuple[RoutingResult, torch.Tensor]:
        """Mode A: Dense Reflected Routing."""
        # ── §1 line 66-73: routing distribution p ──
        if self.cfg.gate_function == "sigmoid":
            # DeepSeek-V3 style: independent sigmoid gates, then normalize.
            # Each expert is gated independently: p_e = σ(s_e / τ)
            # Pressure penalty directly shifts sigmoid input, cleanly
            # deactivating overloaded experts without competitive softmax.
            p_raw = torch.sigmoid(s / self.cfg.temperature)  # [T, E]
            p = p_raw / p_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            p = torch.softmax(s / self.cfg.temperature, dim=-1)  # [T, E]

        # ── §2 line 92-93: routed mass m_e = Σ_t p_{t,e} ──
        m = p.detach().sum(dim=0)  # [E]

        # All experts active for all tokens
        indices = torch.arange(E, device=device).unsqueeze(0).expand(T, -1)  # [T, E]
        dispatch_mask = torch.ones(T, E, dtype=torch.bool, device=device)
        token_ranks = torch.zeros(T, E, dtype=torch.long, device=device)

        load_fraction = m / m.sum().clamp_min(1e-8)
        diagnostics = RoutingDiagnostics(
            raw_load=m,
            load=m,
            load_fraction=load_fraction,
            raw_load_fraction=load_fraction,
            capacity=torch.full((E,), float(T), device=device),
            dropped=torch.zeros((), device=device),
            dropped_assignments=torch.zeros((), device=device),
            entropy=routing_entropy(p),
            overflow=torch.zeros(E, device=device),
            aux_loss=torch.zeros((), device=device),
            capacity_utilization=torch.ones((), device=device),
            matched_compute_fraction=torch.ones((), device=device),
            pressure=self.pressure.q.detach().clone(),
            extra={"indices": indices, "dispatch_mask": dispatch_mask},
        )
        return RoutingResult(indices, p, dispatch_mask, token_ranks, diagnostics), m

    def _route_sparse(
        self, s: torch.Tensor, T: int, E: int, device: torch.device
    ) -> tuple[RoutingResult, torch.Tensor]:
        """Mode B: Sparse Reflected Deployment."""
        # Select top-k and compute combine weights
        if self.cfg.gate_function == "sigmoid":
            # DeepSeek-V3 style: sigmoid → top-k → renormalize
            p_sigmoid = torch.sigmoid(s / self.cfg.temperature)  # [T, E]
            top_weights, indices = torch.topk(p_sigmoid, k=self.cfg.top_k, dim=-1)
            raw_combine_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            priorities = top_weights
        else:
            top_scores, indices = torch.topk(s, k=self.cfg.top_k, dim=-1)  # [T, K]
            raw_combine_weights = torch.softmax(top_scores / self.cfg.temperature, dim=-1)  # [T, K]
            priorities = top_scores

        # Apply capacity policy constraints
        dispatch_mask, load, raw_load, overflow, token_ranks = self.capacity.enforce(
            indices,
            priorities=priorities,
        )
        # Sparse capacity is slot-based, so pressure must see selected-slot demand.
        # Scaling by K keeps total pressure mass equal to T, matching dense mode.
        m = raw_load.detach().to(dtype=s.dtype) / float(self.cfg.top_k)
        combine_weights = raw_combine_weights * dispatch_mask.to(raw_combine_weights.dtype)

        # Renormalize to sum to 1 over active assignments
        denom = combine_weights.sum(dim=-1, keepdim=True)
        combine_weights = torch.where(denom > 0, combine_weights / denom.clamp_min(1e-8), combine_weights)

        # Calculate metrics using fractional loads
        load_fraction = load.float() / load.sum().clamp_min(1)
        raw_load_fraction = raw_load.float() / raw_load.sum().clamp_min(1)

        capacity = self.capacity.capacity(T, device)
        accepted_assignments = load.sum().float()
        requested_assignments = raw_load.sum().float().clamp_min(1.0)

        # Full distribution for entropy diagnostic
        if self.cfg.gate_function == "sigmoid":
            _p_full = torch.sigmoid(s / self.cfg.temperature)
            _p_entropy = _p_full / _p_full.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            _p_entropy = torch.softmax(s / self.cfg.temperature, dim=-1)

        diagnostics = RoutingDiagnostics(
            raw_load=raw_load,
            load=load,
            load_fraction=load_fraction,
            raw_load_fraction=raw_load_fraction,
            capacity=capacity,
            dropped=(~dispatch_mask).all(dim=-1).float().mean(),
            dropped_assignments=(~dispatch_mask).float().mean(),
            entropy=routing_entropy(_p_entropy),
            overflow=overflow,
            aux_loss=torch.zeros((), device=device),
            capacity_utilization=accepted_assignments / (capacity.sum().clamp_min(1.0)),
            matched_compute_fraction=accepted_assignments / requested_assignments,
            pressure=self.pressure.q.detach().clone(),
            extra={"indices": indices, "dispatch_mask": dispatch_mask},
        )

        return RoutingResult(indices, combine_weights, dispatch_mask, token_ranks, diagnostics), m

    def pressure_state_dict(self) -> dict[str, torch.Tensor]:
        return self.pressure.state_dict()

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        if state is not None:
            self.pressure.load_state_dict(state)
            self.pressure.q.clamp_(min=0.0)
