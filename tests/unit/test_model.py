from __future__ import annotations

import torch

from moe_route.models.transformer import DecoderOnlyLM, ModelConfig
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
        router=RouterConfig(kind="reflected", d_model=32, num_experts=4, top_k=2),
    )
    model = DecoderOnlyLM(cfg)
    input_ids = torch.randint(0, 259, (2, 16))
    _, loss, parts = model(input_ids, input_ids)
    assert loss is not None
    assert torch.isfinite(loss)
    assert "aux_loss" in parts
