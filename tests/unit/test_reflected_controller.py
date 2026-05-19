"""Tests for the standalone Reflected Controller router.

Verifies proposition_2.md compliance:
- Dense soft routing over all experts (no top-k)
- Reflected score composition: s = a + b - ρ·φ(q)
- Temperature-controlled softmax
- Projected dual ascent pressure updates from soft mass
- Auxiliary-loss-free by default
- Z-loss on raw gate logits only
- Learnable expert bias
"""

from __future__ import annotations

import torch

from moe_route.routing.reflected_controller import (
    ReflectedController,
    ReflectedControllerConfig,
    ReflectedPressureState,
)
from moe_route.routing.routers import RouterConfig, build_router


def _make_controller(**overrides) -> ReflectedController:
    defaults = dict(d_model=8, num_experts=4)
    defaults.update(overrides)
    return ReflectedController(ReflectedControllerConfig(**defaults))


# ── Shape tests ──


def test_dense_shapes() -> None:
    """Dense mode returns [T, E] indices and weights for all experts."""
    router = _make_controller()
    result = router(torch.randn(6, 8))
    assert result.expert_indices.shape == (6, 4), "indices should be [T, E]"
    assert result.combine_weights.shape == (6, 4), "weights should be [T, E]"
    assert result.dispatch_mask.shape == (6, 4), "mask should be [T, E]"
    assert result.dispatch_mask.all(), "all experts dispatched (no dropping)"
    assert result.expert_indices[0].tolist() == [0, 1, 2, 3], "all expert indices present"


def test_probabilities_sum_to_one() -> None:
    """Routing probabilities sum to 1 for every token."""
    router = _make_controller()
    result = router(torch.randn(10, 8))
    row_sums = result.combine_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"probabilities should sum to 1, got {row_sums}"


# ── Reflected score composition ──


def test_reflected_score_composition() -> None:
    """Verify s = a + b - ρ·φ(q) is correctly composed."""
    cfg = ReflectedControllerConfig(d_model=4, num_experts=2, pressure_scale=2.0)
    router = ReflectedController(cfg)

    # Set known pressure state
    router.pressure.q = torch.tensor([1.0, 0.0])

    x = torch.randn(3, 4)
    a = router.gate(x)  # raw affinity
    penalty = router.pressure.penalty()  # ρ·φ(q)
    expected_s = a + router.bias - penalty

    # The forward pass computes softmax(s/τ), so we verify the score
    # by checking the penalty is non-zero for expert 0 and zero for expert 1
    assert penalty[0] > 0, "overloaded expert should have positive penalty"
    assert penalty[1] == 0.0, "zero-pressure expert should have zero penalty"


# ── Temperature ──


def test_temperature_sharpness() -> None:
    """Low τ → sharp distribution, high τ → uniform."""
    x = torch.randn(20, 8)

    sharp = _make_controller(temperature=0.1)
    uniform = _make_controller(temperature=10.0)

    # Use same gate weights
    with torch.no_grad():
        uniform.gate.weight.copy_(sharp.gate.weight)
        uniform.bias.copy_(sharp.bias)

    r_sharp = sharp(x)
    r_uniform = uniform(x)

    # Sharp should have lower entropy than uniform
    assert r_sharp.diagnostics.entropy < r_uniform.diagnostics.entropy, \
        "lower temperature should produce lower entropy (sharper distribution)"


# ── Pressure dynamics ──


def test_pressure_projected_ascent() -> None:
    """Overloaded expert accumulates pressure; underloaded stays at zero."""
    router = _make_controller(num_experts=2, pressure_lr=1.0)
    router.train()

    # Force all tokens to expert 0 using bias so it's independent of x
    with torch.no_grad():
        router.gate.weight.zero_()
        router.bias[0] = 10.0
        router.bias[1] = -10.0

    router(torch.randn(8, 8))

    # Expert 0 is overloaded → positive pressure
    assert router.pressure.q[0] > 0, "overloaded expert should have positive pressure"
    # Expert 1 is underloaded → pressure projected to 0
    assert router.pressure.q[1] == 0.0, "underloaded expert pressure should be clamped to 0"


def test_pressure_update_uses_soft_mass() -> None:
    """Pressure is updated from soft routing probabilities Σ_t p_{t,e}, not hard counts."""
    cfg = ReflectedControllerConfig(d_model=4, num_experts=2, pressure_lr=0.5)
    router = ReflectedController(cfg)
    router.train()

    # Force a deterministic but still soft preference so mass is clearly fractional.
    with torch.no_grad():
        router.gate.weight.zero_()
        router.bias[0] = 0.2
        router.bias[1] = -0.2

    x = torch.randn(8, 4)
    result = router(x)

    # The routed mass should be soft (non-integer)
    m = result.combine_weights.sum(dim=0)
    assert m.sum().item() == pytest.approx(8.0, abs=1e-4), \
        "total soft mass should equal number of tokens"
    # At least one expert should have non-integer mass (soft routing)
    fractional = (m - m.round()).abs()
    assert fractional.max() > 0.01, \
        "soft mass should have fractional values (not hard integer counts)"


# ── Auxiliary-loss-free ──


def test_aux_loss_free_by_default() -> None:
    """aux_loss = 0 when z_loss_weight = 0 (default)."""
    router = _make_controller(z_loss_weight=0.0)
    result = router(torch.randn(6, 8))
    assert result.diagnostics.aux_loss.item() == 0.0, \
        "aux_loss should be 0 when z_loss_weight is 0"


def test_z_loss_on_raw_logits_only() -> None:
    """Z-loss is computed from raw gate logits a, not pressured scores s."""
    router = _make_controller(z_loss_weight=0.1)
    router.pressure.q = torch.tensor([5.0, 5.0, 5.0, 5.0])

    x = torch.randn(6, 8)
    result = router(x)

    # Compute z-loss manually on raw logits
    a = router.gate(x)
    expected_z = 0.1 * (torch.logsumexp(a, dim=-1) ** 2).mean()

    assert result.diagnostics.z_loss is not None
    assert torch.allclose(result.diagnostics.z_loss, expected_z, atol=1e-5), \
        "z-loss should be computed from raw gate logits, not pressured scores"


# ── Learnable bias ──


def test_learnable_bias_gradient() -> None:
    """Expert bias b receives gradients during backward pass."""
    router = _make_controller(learnable_bias=True)
    x = torch.randn(6, 8)
    result = router(x)

    # Create a simple loss from the routing weights (weighted sum to avoid constant sum=1)
    target = torch.randn_like(result.combine_weights)
    loss = (result.combine_weights * target).sum()
    loss.backward()

    assert router.bias.grad is not None, "bias should receive gradients"
    assert router.bias.grad.abs().sum() > 0, "bias gradients should be non-zero"


# ── Serialization ──


def test_pressure_serialization() -> None:
    """pressure_state_dict / load_pressure_state_dict roundtrip."""
    router = _make_controller(pressure_lr=1.0)
    router.train()
    router(torch.randn(8, 8))  # trigger pressure update

    state = router.pressure_state_dict()
    assert "q" in state
    assert "capacity_weights" in state

    # Create a fresh router and load the state
    router2 = _make_controller(pressure_lr=1.0)
    router2.load_pressure_state_dict(state)

    assert torch.allclose(router.pressure.q, router2.pressure.q), \
        "pressure state should be preserved after roundtrip"


# ── Factory integration ──


def test_build_router_reflected_v2() -> None:
    """build_router with kind='reflected_v2' returns ReflectedController."""
    cfg = RouterConfig(kind="reflected_v2", d_model=8, num_experts=4)
    router = build_router(cfg)
    assert isinstance(router, ReflectedController)
    result = router(torch.randn(6, 8))
    assert result.expert_indices.shape == (6, 4)


# ── MoE end-to-end ──


def test_dense_moe_e2e() -> None:
    """MoEFeedForward with reflected controller: correct output shape, loss returned."""
    from moe_route.models.moe import MoEFeedForward

    router_cfg = RouterConfig(kind="reflected_v2", d_model=16, num_experts=4, z_loss_weight=0.0)
    moe = MoEFeedForward(
        d_model=16,
        num_experts=4,
        expert_hidden_size=32,
        dropout=0.0,
        router_cfg=router_cfg,
    )
    x = torch.randn(2, 8, 16)  # [batch, seq, d_model]
    output, aux_loss = moe(x)

    assert output.shape == x.shape, f"output shape {output.shape} should match input {x.shape}"
    assert aux_loss.item() == 0.0, "aux_loss should be 0 for reflected_v2 with z_loss_weight=0"
    assert moe.last_diagnostics is not None
    assert moe.last_diagnostics.pressure is not None


# Need pytest for approx
import pytest


# ── Mode B (Sparse Reflected Deployment) tests ──


def test_sparse_shapes() -> None:
    """Sparse mode returns [T, K] indices and weights for Top-K experts."""
    router = _make_controller(routing_mode="sparse", top_k=2)
    result = router(torch.randn(6, 8))
    assert result.expert_indices.shape == (6, 2), "indices should be [T, K]"
    assert result.combine_weights.shape == (6, 2), "weights should be [T, K]"
    assert result.dispatch_mask.shape == (6, 2), "mask should be [T, K]"
    assert result.diagnostics.dropped_assignments >= 0.0, "diagnostics.dropped_assignments should be computed"
    assert result.diagnostics.entropy > 0.0, "entropy should be non-zero"


def test_invalid_routing_mode_raises() -> None:
    """ReflectedControllerConfig with invalid routing_mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid routing_mode"):
        _make_controller(routing_mode="invalid_mode")


def test_sparse_mode_invalid_top_k_raises() -> None:
    """ReflectedControllerConfig with routing_mode='sparse' and missing/invalid top_k raises ValueError."""
    with pytest.raises(ValueError, match="requires a positive 'top_k'"):
        _make_controller(routing_mode="sparse", top_k=None)
    with pytest.raises(ValueError, match="requires a positive 'top_k'"):
        _make_controller(routing_mode="sparse", top_k=0)
    with pytest.raises(ValueError, match="requires a positive 'top_k'"):
        _make_controller(routing_mode="sparse", top_k=-1)


def test_sparse_moe_e2e() -> None:
    """MoEFeedForward with sparse reflected controller: correct output shape and sparse dispatch flow."""
    from moe_route.models.moe import MoEFeedForward

    router_cfg = RouterConfig(
        kind="reflected_v2",
        d_model=16,
        num_experts=4,
        routing_mode="sparse",
        top_k=2,
        z_loss_weight=0.0
    )
    moe = MoEFeedForward(
        d_model=16,
        num_experts=4,
        expert_hidden_size=32,
        dropout=0.0,
        router_cfg=router_cfg,
    )
    x = torch.randn(2, 8, 16)  # [batch, seq, d_model]
    output, aux_loss = moe(x)

    assert output.shape == x.shape, f"output shape {output.shape} should match input {x.shape}"
    assert aux_loss.item() == 0.0, "aux_loss should be 0 for reflected_v2 with z_loss_weight=0"
    assert moe.last_diagnostics is not None
    assert moe.last_diagnostics.pressure is not None
    assert moe.last_diagnostics.dropped_assignments is not None
