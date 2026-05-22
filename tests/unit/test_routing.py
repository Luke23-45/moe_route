from __future__ import annotations

import torch

from moe_route.routing.capacity import CapacityPolicy
from moe_route.routing.routers import RouterConfig, build_router


def test_capacity_prefers_high_priority_assignments() -> None:
    policy = CapacityPolicy(num_experts=1, top_k=1, capacity_factor=0.5)
    indices = torch.zeros(4, 1, dtype=torch.long)
    priorities = torch.tensor([[0.1], [0.9], [0.2], [0.8]])

    dispatch_mask, load, raw_load, overflow, token_ranks = policy.enforce(
        indices,
        priorities=priorities,
    )

    assert dispatch_mask.squeeze(-1).tolist() == [False, True, False, True]
    assert token_ranks.squeeze(-1).tolist() == [3, 0, 2, 1]
    assert load.tolist() == [2]
    assert raw_load.tolist() == [4]
    assert overflow.tolist() == [2]


def test_topk_router_shapes_and_capacity() -> None:
    router = build_router(
        RouterConfig(kind="topk", d_model=8, num_experts=4, top_k=2, capacity_factor=1.0)
    )
    result = router(torch.randn(6, 8))
    assert result.expert_indices.shape == (6, 2)
    assert result.combine_weights.shape == (6, 2)
    assert result.dispatch_mask.shape == (6, 2)
    assert result.diagnostics.load.shape == (4,)
    assert result.diagnostics.raw_load.shape == (4,)
    assert result.diagnostics.raw_load.sum() == 12
    assert 0.0 <= float(result.diagnostics.matched_compute_fraction) <= 1.0


def test_reflected_router_updates_pressure() -> None:
    router = build_router(
        RouterConfig(kind="reflected_v2", routing_mode="sparse", d_model=8, num_experts=4, top_k=1, pressure_lr=0.5)
    )
    router.train()
    result = router(torch.randn(16, 8))
    assert result.diagnostics.pressure is not None
    assert torch.all(result.diagnostics.pressure >= 0)


def test_reflected_router_uses_raw_load_for_pressure() -> None:
    router = build_router(
        RouterConfig(kind="reflected_v2", routing_mode="sparse", d_model=4, num_experts=2, top_k=1, capacity_factor=0.1)
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0].fill_(1.0)
        router.gate.weight[1].fill_(-1.0)
    router.train()
    result = router(torch.ones(8, 4))
    assert result.diagnostics.raw_load[0] == 8
    assert result.diagnostics.load[0] < result.diagnostics.raw_load[0]
    assert result.diagnostics.pressure is not None
    assert result.diagnostics.pressure[0] > result.diagnostics.pressure[1]
