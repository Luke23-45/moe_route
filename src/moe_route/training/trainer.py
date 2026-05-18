from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from moe_route.data.pipeline import build_dataloader
from moe_route.data.prepare import prepare_data
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.tracking.factory import build_tracker
from moe_route.training.checkpoint import load_checkpoint, save_checkpoint
from moe_route.training.distributed import cleanup_distributed, init_distributed, wrap_model
from moe_route.training.optim import build_optimizer, build_scheduler
from moe_route.utils.env import write_run_metadata
from moe_route.utils.progress import training_bar
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
        metrics[f"{prefix}/dropped_assignments"] = float(diag.dropped_assignments.detach().cpu())
        metrics[f"{prefix}/entropy"] = float(diag.entropy.detach().cpu())
        metrics[f"{prefix}/overflow"] = float(diag.overflow.float().sum().detach().cpu())
        metrics[f"{prefix}/capacity_utilization"] = float(diag.capacity_utilization.detach().cpu())
        metrics[f"{prefix}/matched_compute_fraction"] = float(
            diag.matched_compute_fraction.detach().cpu()
        )
        if diag.pressure is not None:
            metrics[f"{prefix}/pressure_mean"] = float(diag.pressure.float().mean().detach().cpu())
    return metrics


def train(cfg) -> Path | None:
    seed_everything(int(cfg.seed))
    ctx = init_distributed(cfg)
    run_dir = Path(cfg.artifact_dir) / "runs" / str(cfg.run_name)
    if ctx.is_main:
        write_run_metadata(run_dir, cfg)
        print(
            f"[train] run={cfg.run_name} device={ctx.device} world_size={ctx.world_size} "
            f"precision={cfg.trainer.precision}"
        )
        if ctx.device.type != "cuda":
            print(
                "[train] CUDA is not available. Training will run on CPU, which is not the intended "
                "path for full TinyStories experiments."
            )
    tracker = build_tracker(cfg, run_dir) if ctx.is_main else None

    tokenizer = build_tokenizer(cfg.tokenizer)
    if bool(cfg.data.get("cache_tokenized", False)) and bool(cfg.trainer.get("prepare_data", True)):
        if ctx.is_main:
            prepare_data(cfg.data, tokenizer, show_progress=True, build_missing=True)
        if ctx.enabled and dist.is_initialized():
            dist.barrier()
    dataloader = build_dataloader(
        cfg.data,
        tokenizer,
        distributed=ctx.enabled,
        rank=ctx.rank,
        world_size=ctx.world_size,
        prepare=not bool(cfg.data.get("cache_tokenized", False)),
    )
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
    steps_per_epoch = max(len(dataloader) // grad_accum, 1)
    configured_epochs = cfg.trainer.get("max_epochs")
    max_epochs = (
        int(configured_epochs)
        if configured_epochs is not None
        else max((max_steps + steps_per_epoch - 1) // steps_per_epoch, 1)
    )
    use_amp = str(cfg.trainer.precision) in {"bf16", "fp16"} and ctx.device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg.trainer.precision) == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    last_ckpt: Path | None = None
    started = time.perf_counter()

    try:
        step = start_step
        for epoch in range(1, max_epochs + 1):
            if step >= max_steps:
                break
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            remaining_steps = max_steps - step
            epoch_steps = min(steps_per_epoch, remaining_steps)
            data_iter = iter(dataloader)
            progress = training_bar(cfg, epoch=epoch, total=epoch_steps) if ctx.is_main else None
            try:
                for _epoch_step in range(epoch_steps):
                    step += 1
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = torch.zeros((), device=ctx.device)
                    for _ in range(grad_accum):
                        try:
                            input_ids, labels = next(data_iter)
                        except StopIteration:
                            data_iter = iter(dataloader)
                            input_ids, labels = next(data_iter)
                        input_ids = input_ids.to(ctx.device, non_blocking=True)
                        labels = labels.to(ctx.device, non_blocking=True)
                        with torch.autocast(
                            device_type=ctx.device.type, dtype=amp_dtype, enabled=use_amp
                        ):
                            _, loss, _ = model(input_ids, labels)
                        if loss is None:
                            raise RuntimeError("Training loss was not produced.")
                        scaler.scale(loss / grad_accum).backward()
                        total_loss = total_loss + loss.detach() / grad_accum

                    if cfg.trainer.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.trainer.clip_grad_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    elapsed = max(time.perf_counter() - started, 1e-6)
                    tokens = step * int(cfg.data.batch_size) * int(cfg.data.sequence_length) * grad_accum
                    metrics = {
                        "train/loss": float(total_loss.detach().cpu()),
                        "train/lr": float(scheduler.get_last_lr()[0]),
                        "perf/tokens_per_sec": tokens / elapsed,
                    }
                    routing_metrics = _routing_metrics(model)
                    metrics.update(routing_metrics)

                    if progress is not None:
                        postfix = {
                            "step": step,
                            "loss": f"{metrics['train/loss']:.3f}",
                            "lr": f"{metrics['train/lr']:.2e}",
                            "tok/s": f"{metrics['perf/tokens_per_sec']:.0f}",
                        }
                        if "router/0/matched_compute_fraction" in routing_metrics:
                            postfix["compute"] = (
                                f"{routing_metrics['router/0/matched_compute_fraction']:.2f}"
                            )
                        progress.set_postfix(postfix, refresh=False)
                        progress.update(1)

                    if tracker is not None and step % int(cfg.trainer.log_every) == 0:
                        tracker.log_metrics(metrics, step)

                    if (
                        ctx.is_main
                        and int(cfg.trainer.checkpoint_every) > 0
                        and step % int(cfg.trainer.checkpoint_every) == 0
                    ):
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
                if progress is not None:
                    progress.close()
    finally:
        if tracker is not None:
            tracker.close()
        cleanup_distributed(ctx)

    return last_ckpt
