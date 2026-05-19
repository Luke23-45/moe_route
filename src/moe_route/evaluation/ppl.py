from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from moe_route.data.pipeline import build_dataloader
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.training.checkpoint import load_checkpoint


@torch.no_grad()
def evaluate_model_perplexity(model, dataloader, max_batches: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_aux = 0.0
    aux_batches = 0
    for idx, (input_ids, labels) in enumerate(dataloader):
        if max_batches > 0 and idx >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, _, parts = model(input_ids, labels)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.detach().cpu())
        total_tokens += int(labels.numel())
        if isinstance(parts, dict) and "aux_loss" in parts:
            total_aux += float(parts["aux_loss"].detach().cpu())
            aux_batches += 1
    mean_loss = total_nll / max(total_tokens, 1)
    metrics = {"eval/loss": mean_loss, "eval/ppl": math.exp(min(mean_loss, 20.0))}
    if aux_batches:
        metrics["eval/aux_loss"] = total_aux / aux_batches
    return metrics


def build_eval_data_cfg(cfg):
    eval_data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    eval_data_cfg.prepared_split = str(eval_data_cfg.get("prepared_split", "train"))
    split_override = cfg.eval.get("data_split", None)
    if split_override is not None:
        eval_data_cfg.split = str(split_override)
        eval_data_cfg.prepared_split = "validation"
        return eval_data_cfg

    if str(eval_data_cfg.get("adapter", "")) != "huggingface":
        return eval_data_cfg
    if str(eval_data_cfg.get("split", "")) != "train":
        return eval_data_cfg

    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(
            str(eval_data_cfg.dataset_name),
            name=None if eval_data_cfg.get("dataset_config") is None else str(eval_data_cfg.get("dataset_config")),
        )
        available_splits = set(builder.info.splits.keys())
    except Exception:
        available_splits = set()

    for candidate in ("validation", "valid", "test"):
        if candidate in available_splits:
            eval_data_cfg.split = candidate
            eval_data_cfg.prepared_split = "validation"
            return eval_data_cfg
    if float(eval_data_cfg.get("holdout_fraction", 0.0)) > 0.0:
        eval_data_cfg.prepared_split = "validation"
    return eval_data_cfg


@torch.no_grad()
def evaluate_perplexity(cfg, checkpoint: str | Path | None = None) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg.tokenizer)
    dataloader = build_dataloader(build_eval_data_cfg(cfg), tokenizer)
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(device)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model)
        
    max_batches = int(cfg.eval.max_batches)
    return evaluate_model_perplexity(model, dataloader, max_batches, device)


def load_cfg_from_checkpoint(checkpoint: str | Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return OmegaConf.create(payload["config"])
