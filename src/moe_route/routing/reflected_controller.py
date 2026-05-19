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


# ---------------------------------------------------------------------------
# Reflected Pressure State (self-contained)
# ---------------------------------------------------------------------------

class ReflectedPressureState:
    """Nonnegative expert-pressure state with projected dual ascent.

    §2: q_e^(n+1) = max(0, q_e^(n) + η·(m_e - c_e))

    The pressure penalty used in routing scores is (§1):
        φ_e(q_e) = q_e / (μ_e^β + ε)

    where μ_e is the per-expert target capacity weight (uniform by default).
    """

    def __init__(self, cfg: ReflectedControllerConfig, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        # Uniform capacity weights: μ_e = 1/E  (§1 line 41-44)
        self.capacity_weights = torch.full(
            (cfg.num_experts,), 1.0 / cfg.num_experts,
            dtype=torch.float32, device=self.device,
        )
        # Pressure state: q ∈ ℝ₊^E, initialized to zero
        self.q = torch.zeros(cfg.num_experts, dtype=torch.float32, device=self.device)

    def to(self, device: torch.device | str) -> ReflectedPressureState:
        self.device = torch.device(device)
        self.q = self.q.to(self.device)
        self.capacity_weights = self.capacity_weights.to(self.device)
        return self

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
        m = mass.detach().to(self.q.device, dtype=self.q.dtype)
        target = num_tokens / self.cfg.num_experts  # c_e = T/E
        delta = m - target

        if self.cfg.pressure_decay > 0:
            self.q.mul_(1.0 - self.cfg.pressure_decay)

        self.q.add_(self.cfg.pressure_lr * delta)
        # §2 line 110: Π_{[0,∞)}(z) = max{0, z}
        self.q.clamp_(min=0.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "q": self.q.detach().clone(),
            "capacity_weights": self.capacity_weights.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.q = state["q"].detach().clone().to(self.device)
        self.capacity_weights = state["capacity_weights"].detach().clone().to(self.device)
        self.q.clamp_(min=0.0)

    def reset(self) -> None:
        self.q.zero_()


# ---------------------------------------------------------------------------
# Reflected Controller Router
# ---------------------------------------------------------------------------

class ReflectedController(nn.Module):
    """Standalone Reflected Controller MoE Router.

    Computes dense soft routing over ALL experts with reflected pressure
    dynamics for load balancing. No top-k, no capacity enforcement,
    no token dropping, no auxiliary balance loss.

    Forward pass (proposition §1-§4):
        1. a = G(x)                              raw affinity
        2. s = a + b − ρ·φ(q)                    reflected score
        3. p = softmax(s / τ)                     routing distribution
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

        # Gate network G_ℓ: ℝ^d → ℝ^E  (§1 line 28-30)
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        # Expert bias b_ℓ ∈ ℝ^E  (§1 line 32-34)
        if cfg.learnable_bias:
            self.bias = nn.Parameter(torch.zeros(cfg.num_experts))
        else:
            self.register_buffer("bias", torch.zeros(cfg.num_experts))

        # Reflected pressure state  (§1 line 36-38)
        self.pressure = ReflectedPressureState(cfg)

        # CUDA async stream for pressure updates
        self._update_stream: torch.cuda.Stream | None = None

    def forward(self, x: torch.Tensor) -> RoutingResult:
        """Route tokens to all experts with reflected soft routing.

        Args:
            x: [T, d_model] flattened token representations.

        Returns:
            RoutingResult with dense routing over all E experts.
        """
        T = x.shape[0]
        E = self.cfg.num_experts
        device = x.device

        # Move pressure state to the correct device
        self.pressure.to(device)

        # ── §1 line 48-49: raw affinity a = G(x) ──
        a = self.gate(x)  # [T, E]

        # ── §1 line 53-57: reflected score s = a + b − ρ·φ(q) ──
        penalty = self.pressure.penalty()  # [E]
        s = a + self.bias - penalty  # [T, E]  (bias and penalty broadcast over T)

        # ── §1 line 66-73: routing distribution p = softmax(s / τ) ──
        p = torch.softmax(s / self.cfg.temperature, dim=-1)  # [T, E]

        # ── §2 line 92-93: routed mass m_e = Σ_t p_{t,e} ──
        m = p.detach().sum(dim=0)  # [E]

        # ── §2 line 96-107: projected reflected ascent (training only) ──
        if self.training:
            if device.type == "cuda":
                if self._update_stream is None:
                    self._update_stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(self._update_stream):
                    self.pressure.update_from_mass(m, T)
            else:
                self.pressure.update_from_mass(m, T)

        # ── §4 line 176-186: z-loss on RAW gate logits a, NOT pressured s ──
        z_loss = torch.zeros((), device=device)
        if self.cfg.z_loss_weight > 0:
            z_loss = self.cfg.z_loss_weight * (torch.logsumexp(a, dim=-1) ** 2).mean()

        # ── aux_loss is z_loss only (0 by default = auxiliary-loss-free, §4) ──
        aux_loss = z_loss

        # ── Build routing result: all experts for every token ──
        indices = torch.arange(E, device=device).unsqueeze(0).expand(T, -1)  # [T, E]
        dispatch_mask = torch.ones(T, E, dtype=torch.bool, device=device)
        token_ranks = torch.zeros(T, E, dtype=torch.long, device=device)

        # ── Diagnostics ──
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
            aux_loss=aux_loss,
            capacity_utilization=torch.ones((), device=device),
            matched_compute_fraction=torch.ones((), device=device),
            z_loss=z_loss if self.cfg.z_loss_weight > 0 else None,
            pressure=self.pressure.q.detach().clone(),
        )

        return RoutingResult(indices, p, dispatch_mask, token_ranks, diagnostics)

    def pressure_state_dict(self) -> dict[str, torch.Tensor]:
        return self.pressure.state_dict()

    def load_pressure_state_dict(self, state: dict[str, torch.Tensor] | None) -> None:
        if state is not None:
            self.pressure.load_state_dict(state)
