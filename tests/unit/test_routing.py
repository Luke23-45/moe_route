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


def test_reflected_router_updates_pressure() -> None:
    router = build_router(
        RouterConfig(kind="reflected", d_model=8, num_experts=4, top_k=1, pressure_lr=0.5)
    )
    router.train()
    result = router(torch.randn(16, 8))
    assert result.diagnostics.pressure is not None
    assert torch.all(result.diagnostics.pressure >= 0)

