from __future__ import annotations

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from moe_route.data.pipeline import build_dataloader
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.training.checkpoint import load_checkpoint


@torch.no_grad()
def evaluate_perplexity(cfg, checkpoint: str | Path | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg.tokenizer)
    dataloader = build_dataloader(cfg.data, tokenizer)
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(device)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model)
    model.eval()

    losses: list[float] = []
    max_batches = int(cfg.eval.max_batches)
    for idx, (input_ids, labels) in enumerate(dataloader):
        if max_batches > 0 and idx >= max_batches:
            break
        _, loss, _ = model(input_ids.to(device), labels.to(device))
        if loss is not None:
            losses.append(float(loss.detach().cpu()))
    mean_loss = sum(losses) / max(len(losses), 1)
    return {"eval/loss": mean_loss, "eval/ppl": math.exp(min(mean_loss, 20.0))}


def load_cfg_from_checkpoint(checkpoint: str | Path):
    payload = torch.load(checkpoint, map_location="cpu")
    return OmegaConf.create(payload["config"])

