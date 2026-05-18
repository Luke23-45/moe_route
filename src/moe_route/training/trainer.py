from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path

import torch
from omegaconf import OmegaConf

from moe_route.data.pipeline import build_dataloader
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.tracking.factory import build_tracker
from moe_route.training.checkpoint import load_checkpoint, save_checkpoint
from moe_route.training.distributed import cleanup_distributed, init_distributed, wrap_model
from moe_route.training.optim import build_optimizer, build_scheduler
from moe_route.utils.env import write_run_metadata
from moe_route.utils.seed import seed_everything


def _routing_metrics(model: torch.nn.Module) -> dict[str, float]:
    raw = model.module if hasattr(model, "module") else model
    diagnostics = raw.routing_diagnostics() if hasattr(raw, "routing_diagnostics") else []
    if not diagnostics:
        return {}
    metrics: dict[str, float] = {}
    for i, diag in enumerate(diagnostics):
        prefix = f"router/{i}"
        metrics[f"{prefix}/drop_rate"] = float(diag.dropped.detach().cpu())
        metrics[f"{prefix}/entropy"] = float(diag.entropy.detach().cpu())
        metrics[f"{prefix}/overflow"] = float(diag.overflow.float().sum().detach().cpu())
        if diag.pressure is not None:
            metrics[f"{prefix}/pressure_mean"] = float(diag.pressure.float().mean().detach().cpu())
    return metrics


def train(cfg) -> Path | None:
    seed_everything(int(cfg.seed))
    ctx = init_distributed(cfg)
    run_dir = Path(cfg.artifact_dir) / "runs" / str(cfg.run_name)
    if ctx.is_main:
        write_run_metadata(run_dir, cfg)
    tracker = build_tracker(cfg, run_dir) if ctx.is_main else None

    tokenizer = build_tokenizer(cfg.tokenizer)
    dataloader = build_dataloader(cfg.data, tokenizer)
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(ctx.device)
    if bool(cfg.trainer.compile):
        model = torch.compile(model)
    model = wrap_model(model, ctx, bool(cfg.distributed.find_unused_parameters))
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_step = 0
    if cfg.trainer.resume_from:
        start_step = load_checkpoint(cfg.trainer.resume_from, model, optimizer, scheduler)

    if tracker is not None:
        tracker.log_params(OmegaConf.to_container(cfg, resolve=True))

    model.train()
    grad_accum = int(cfg.trainer.grad_accum_steps)
    max_steps = int(cfg.trainer.max_steps)
    iterator = cycle(dataloader)
    last_ckpt: Path | None = None
    started = time.perf_counter()

    try:
        for step in range(start_step + 1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=ctx.device)
            for _ in range(grad_accum):
                input_ids, labels = next(iterator)
                input_ids = input_ids.to(ctx.device, non_blocking=True)
                labels = labels.to(ctx.device, non_blocking=True)
                _, loss, parts = model(input_ids, labels)
                if loss is None:
                    raise RuntimeError("Training loss was not produced.")
                (loss / grad_accum).backward()
                total_loss = total_loss + loss.detach() / grad_accum

            if cfg.trainer.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.trainer.clip_grad_norm))
            optimizer.step()
            scheduler.step()

            if tracker is not None and step % int(cfg.trainer.log_every) == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                tokens = step * int(cfg.data.batch_size) * int(cfg.data.sequence_length) * grad_accum
                metrics = {
                    "train/loss": float(total_loss.detach().cpu()),
                    "train/lr": float(scheduler.get_last_lr()[0]),
                    "perf/tokens_per_sec": tokens / elapsed,
                }
                metrics.update(_routing_metrics(model))
                tracker.log_metrics(metrics, step)

            if ctx.is_main and int(cfg.trainer.checkpoint_every) > 0 and step % int(cfg.trainer.checkpoint_every) == 0:
                last_ckpt = save_checkpoint(
                    Path(cfg.trainer.save_dir) / f"step_{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    OmegaConf.to_container(cfg, resolve=True),
                )
                if tracker is not None:
                    tracker.log_artifact(last_ckpt)
    finally:
        if tracker is not None:
            tracker.close()
        cleanup_distributed(ctx)

    return last_ckpt

