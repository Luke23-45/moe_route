from __future__ import annotations

import torch
from omegaconf import OmegaConf

from moe_route.models.transformer import DecoderOnlyLM, ModelConfig, build_model_cfg
from moe_route.routing.routers import RouterConfig


def test_decoder_moe_forward_loss() -> None:
    cfg = ModelConfig(
        vocab_size=259,
        max_seq_len=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        moe_enabled=True,
        moe_every_n_layers=1,
        num_experts=4,
        expert_hidden_size=64,
        router=RouterConfig(kind="reflected_v2", routing_mode="sparse", d_model=32, num_experts=4, top_k=2),
    )
    model = DecoderOnlyLM(cfg)
    input_ids = torch.randint(0, 259, (2, 16))
    _, loss, parts = model(input_ids, input_ids)
    assert loss is not None
    assert torch.isfinite(loss)
    assert "aux_loss" in parts


def test_build_model_cfg_preserves_router_routing_mode() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "vocab_size": 256,
                "max_seq_len": 8,
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 32,
                "dropout": 0.0,
                "moe": {
                    "enabled": True,
                    "every_n_layers": 1,
                    "num_experts": 2,
                    "expert_hidden_size": 32,
                },
            },
            "router": {
                "kind": "reflected_v2",
                "routing_mode": "sparse",
                "top_k": 2,
                "capacity_factor": 1.25,
                "drop_tokens": True,
            },
        }
    )
    model_cfg = build_model_cfg(cfg)
    assert model_cfg.router.routing_mode == "sparse"
