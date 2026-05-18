from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(parameters, cfg) -> Optimizer:
    kwargs = {
        "lr": float(cfg.optimizer.lr),
        "betas": tuple(float(x) for x in cfg.optimizer.betas),
        "weight_decay": float(cfg.optimizer.weight_decay),
        "eps": float(cfg.optimizer.eps),
    }
    if bool(cfg.optimizer.fused) and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def build_scheduler(optimizer: Optimizer, cfg) -> LambdaLR:
    warmup = int(cfg.optimizer.scheduler.warmup_steps)
    total = max(int(cfg.trainer.max_steps), 1)
    min_ratio = float(cfg.optimizer.scheduler.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max((step + 1) / warmup, 1e-8)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)

