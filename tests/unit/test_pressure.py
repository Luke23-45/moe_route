from __future__ import annotations

import torch

from moe_route.routing.pressure import PressureState, PressureStateConfig


def test_pressure_update_reflects_nonnegative() -> None:
    state = PressureState(PressureStateConfig(num_experts=3, lr=1.0))
    state.update(torch.tensor([0.0, 0.0, 1.0]))
    assert torch.all(state.q >= 0)
    assert state.q[2] > 0


def test_pressure_serialization_roundtrip() -> None:
    state = PressureState(PressureStateConfig(num_experts=2))
    state.update(torch.tensor([1.0, 0.0]))
    clone = PressureState(PressureStateConfig(num_experts=2))
    clone.load_state_dict(state.state_dict())
    assert torch.allclose(state.q, clone.q)

