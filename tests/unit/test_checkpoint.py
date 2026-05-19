from __future__ import annotations

from moe_route.models.transformer import DecoderOnlyLM, ModelConfig
from moe_route.routing.routers import RouterConfig
from moe_route.training.checkpoint import load_checkpoint, save_checkpoint
from moe_route.training.optim import build_scheduler


class _Cfg:
    trainer = type("Trainer", (), {"max_steps": 2})()
    optimizer = type(
        "Optimizer",
        (),
        {"scheduler": type("Scheduler", (), {"warmup_steps": 0, "min_lr_ratio": 0.1})()},
    )()


def _model() -> DecoderOnlyLM:
    return DecoderOnlyLM(
        ModelConfig(
            vocab_size=256,
            max_seq_len=8,
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=32,
            dropout=0.0,
            moe_enabled=True,
            moe_every_n_layers=1,
            num_experts=2,
            expert_hidden_size=32,
            router=RouterConfig(kind="reflected_v2", routing_mode="dense", d_model=16, num_experts=2, top_k=1),
        )
    )


def test_checkpoint_roundtrip_restores_step_and_pressure(tmp_path) -> None:
    import torch

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, _Cfg())
    model(torch.randint(0, 255, (2, 8)), torch.randint(0, 255, (2, 8)))
    pressure_state = model.pressure_state_dict()
    pressure_state[0][0]["q"] = torch.tensor([0.25, 0.75])
    model.load_pressure_state_dict(pressure_state)
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, 1, {"seed": 1})

    restored = _model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = build_scheduler(restored_optimizer, _Cfg())
    step = load_checkpoint(path, restored, restored_optimizer, restored_scheduler)

    assert step == 1
    assert torch.allclose(
        restored.pressure_state_dict()[0][0]["q"],
        torch.tensor([0.25, 0.75]),
    )
