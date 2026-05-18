from __future__ import annotations

import torch

from moe_route.routing.routers import RouterConfig, build_router


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
        RouterConfig(kind="reflected", d_model=8, num_experts=4, top_k=1, pressure_lr=0.5)
    )
    router.train()
    result = router(torch.randn(16, 8))
    assert result.diagnostics.pressure is not None
    assert torch.all(result.diagnostics.pressure >= 0)


def test_reflected_router_uses_raw_load_for_pressure() -> None:
    router = build_router(
        RouterConfig(kind="reflected", d_model=4, num_experts=2, top_k=1, capacity_factor=0.1)
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
