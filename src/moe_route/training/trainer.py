from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from moe_route.data.pipeline import build_dataloader
from moe_route.data.prepare import prepare_data
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer, resolve_char_token_ids
from moe_route.tracking.factory import build_tracker
from moe_route.training.checkpoint import load_checkpoint, save_checkpoint
from moe_route.training.distributed import cleanup_distributed, init_distributed, wrap_model
from moe_route.training.optim import build_optimizer, build_scheduler
from moe_route.utils.env import write_run_metadata
from moe_route.utils.progress import training_bar
from moe_route.utils.seed import seed_everything


def _routing_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor | None = None,
    tokenizer=None,
) -> dict[str, float]:
    raw = model.module if hasattr(model, "module") else model
    diagnostics = raw.routing_diagnostics() if hasattr(raw, "routing_diagnostics") else []
    if not diagnostics:
        return {}
    metrics: dict[str, float] = {}
    
    # Pre-classify domains if input_ids is provided.
    # Resolve token IDs from the tokenizer when available; fall back to
    # ByteTokenizer defaults (byte_value + 3) for backward compatibility.
    token_domains = None
    if input_ids is not None:
        if tokenizer is not None:
            try:
                char_ids = resolve_char_token_ids(tokenizer, '".,?!')
                quote_id = char_ids['"']
                punct_ids = [char_ids[c] for c in '.,?!']
            except ValueError:
                # Tokenizer doesn't support single-char encoding; skip domain metrics.
                quote_id = None
                punct_ids = []
        else:
            # Legacy fallback: ByteTokenizer (byte_value + 3).
            quote_id = 37
            punct_ids = [49, 47, 66, 36]

        if quote_id is not None:
            B, S = input_ids.shape
            # Count quotation marks
            quote_counts = (input_ids == quote_id).sum(dim=-1)
            # Count punctuation marks
            punct_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for pid in punct_ids:
                punct_mask |= input_ids == pid
            punct_counts = punct_mask.sum(dim=-1)
            
            # Classify domains (0: normal, 1: dialogue, 2: punctuation)
            is_dialogue = quote_counts >= 4
            is_punct = (punct_counts >= 11) & (~is_dialogue)
            
            domain_labels = torch.zeros(B, dtype=torch.long, device=input_ids.device)
            domain_labels[is_dialogue] = 1
            domain_labels[is_punct] = 2
            
            # Expand to token level: [B * S]
            token_domains = domain_labels.unsqueeze(1).expand(-1, S).reshape(-1)

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
        if diag.z_loss is not None:
            metrics[f"{prefix}/z_loss"] = float(diag.z_loss.detach().cpu())
        if diag.pressure is not None:
            metrics[f"{prefix}/pressure_mean"] = float(diag.pressure.float().mean().detach().cpu())

        # Load Imbalance Metrics (CV and Gini)
        raw_load = diag.raw_load.float()
        E = raw_load.numel()
        mean_raw_load = raw_load.mean()
        if mean_raw_load > 0:
            load_cv = float((raw_load.std() / (mean_raw_load + 1e-8)).detach().cpu())
            # Gini coefficient
            diff = torch.abs(raw_load.unsqueeze(0) - raw_load.unsqueeze(1))
            load_gini = float((diff.sum() / (2.0 * E * raw_load.sum() + 1e-8)).detach().cpu())
        else:
            load_cv = 0.0
            load_gini = 0.0
            
        metrics[f"{prefix}/load_cv"] = load_cv
        metrics[f"{prefix}/load_gini"] = load_gini

        # DESI metric
        if token_domains is not None and "indices" in diag.extra:
            indices = diag.extra["indices"]  # [T, K]
            dispatch_mask = diag.extra.get("dispatch_mask")
            K = indices.shape[-1]
            token_domains_expanded = token_domains.unsqueeze(1).expand(-1, K).reshape(-1)
            flat_indices = indices.reshape(-1)
            if dispatch_mask is not None:
                flat_mask = dispatch_mask.reshape(-1)
                token_domains_expanded = token_domains_expanded[flat_mask]
                flat_indices = flat_indices[flat_mask]

            if flat_indices.numel() == 0:
                metrics[f"{prefix}/desi"] = 0.0
                continue
            
            # Joint distribution counts of shape [3, E]
            counts = torch.zeros(3, E, dtype=torch.float, device=indices.device)
            joint_index = token_domains_expanded * E + flat_indices
            counts.view(-1).index_add_(0, joint_index, torch.ones_like(joint_index, dtype=torch.float))
            
            p_e_given_d = counts / (counts.sum(dim=1, keepdim=True) + 1e-8)
            p_e = counts.sum(dim=0) / (counts.sum() + 1e-8)
            
            ratio = p_e_given_d / (p_e.unsqueeze(0) + 1e-8)
            kl = (p_e_given_d * torch.log(ratio + 1e-8)).sum(dim=1)
            metrics[f"{prefix}/desi"] = float(kl.mean().detach().cpu())

    return metrics


def _progress_postfix(step: int, lr: float, loss_value: float, tokens_per_sec: float, routing_metrics: dict[str, float]) -> dict[str, str | int]:
    postfix: dict[str, str | int] = {
        "step": step,
        "loss": f"{loss_value:.3f}",
        "lr": f"{lr:.2e}",
        "tok/s": f"{tokens_per_sec:.0f}",
    }
    if "router/0/matched_compute_fraction" in routing_metrics:
        postfix["compute"] = f"{routing_metrics['router/0/matched_compute_fraction']:.2f}"
    if "router/0/drop_rate" in routing_metrics:
        postfix["drop"] = f"{routing_metrics['router/0/drop_rate']:.3f}"
    if "router/0/capacity_utilization" in routing_metrics:
        postfix["cap"] = f"{routing_metrics['router/0/capacity_utilization']:.2f}"
    if "router/0/entropy" in routing_metrics:
        postfix["entropy"] = f"{routing_metrics['router/0/entropy']:.2f}"
    if "router/0/load_cv" in routing_metrics:
        postfix["cv"] = f"{routing_metrics['router/0/load_cv']:.2f}"
    if "router/0/desi" in routing_metrics:
        postfix["desi"] = f"{routing_metrics['router/0/desi']:.2f}"
    return postfix


def train(cfg) -> Path | None:
    from moe_route.utils.compile_state import CompileState

    seed_everything(int(cfg.seed))
    ctx = init_distributed(cfg)
    run_dir = Path(cfg.artifact_dir) / "runs" / str(cfg.run_name)
    
    # --- Robust Hardware Checks for SOTA Optimizations ---
    precision = str(cfg.trainer.get("precision", "fp32"))
    compile_requested = bool(cfg.trainer.get("compile", False))
    compile_mode = cfg.trainer.get("compile_mode", None)

    # Initialize the CompileState state machine. This automatically validates GPU support,
    # manages dynamic allocations, and configures Inductor & Memory safety parameters.
    compile_enabled, dynamic_logic_active = CompileState.initialize(
        compile_requested=compile_requested,
        device_type=ctx.device.type,
        is_main=ctx.is_main
    )

    if ctx.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        capability = torch.cuda.get_device_capability()
        # bf16 requires Ampere (8.0) or higher for hardware acceleration
        if precision == "bf16" and (capability[0] < 8 or not torch.cuda.is_bf16_supported()):
            if ctx.is_main:
                print(f"[train] WARNING: Native bfloat16 requires Compute Capability >= 8.0 (found {capability}). Falling back to fp16.")
            precision = "fp16"
    else:
        if precision == "fp16":
            if ctx.is_main:
                print("[train] WARNING: CPU autocast strictly supports bfloat16. Switching fp16 to bf16.")
            precision = "bf16"
            
    try:
        OmegaConf.update(cfg, "trainer.precision", precision, force_add=True)
        OmegaConf.update(cfg, "trainer.compile", compile_enabled, force_add=True)
    except Exception:
        cfg.trainer.precision = precision
        cfg.trainer.compile = compile_enabled
    # -----------------------------------------------------

    if ctx.is_main:
        write_run_metadata(run_dir, cfg)
        print(
            f"[train] run={cfg.run_name} device={ctx.device} world_size={ctx.world_size} "
            f"precision={precision} compile={compile_enabled}"
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
    if compile_enabled:
        compile_kwargs = {}
        if compile_mode is not None:
            compile_kwargs["mode"] = str(compile_mode)
        model = torch.compile(model, **compile_kwargs)
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
    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16 and ctx.device.type == "cuda")
    last_ckpt: Path | None = None
    started = time.perf_counter()

    try:
        step = start_step
        routing_metrics = {}
        latest_postfix: dict[str, str | int] | None = None
        log_every = int(cfg.trainer.log_every)
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
                    total_aux_loss = torch.zeros((), device=ctx.device)
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
                            _, loss, parts = model(input_ids, labels)
                        if loss is None:
                            raise RuntimeError("Training loss was not produced.")
                        scaler.scale(loss / grad_accum).backward()
                        total_loss = total_loss + loss.detach() / grad_accum
                        total_aux_loss = total_aux_loss + parts["aux_loss"] / grad_accum

                    if cfg.trainer.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.trainer.clip_grad_norm)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    should_log = step % log_every == 0 or step == start_step + 1

                    if should_log:
                        elapsed = max(time.perf_counter() - started, 1e-6)
                        tokens = step * int(cfg.data.batch_size) * int(cfg.data.sequence_length) * grad_accum
                        loss_value = float(total_loss.detach().cpu())
                        aux_value = float(total_aux_loss.detach().cpu())
                        lr_value = float(scheduler.get_last_lr()[0])
                        metrics = {
                            "train/loss": loss_value,
                            "train/lm_loss": loss_value - aux_value,
                            "train/aux_loss": aux_value,
                            "train/lr": lr_value,
                            "perf/tokens_per_sec": tokens / elapsed,
                        }
                        # NOTE: When grad_accum > 1, input_ids is from the LAST microbatch
                        # only, so routing diagnostics reflect a single microbatch, not the
                        # full accumulated batch.
                        routing_metrics = _routing_metrics(model, input_ids, tokenizer=tokenizer)
                        metrics.update(routing_metrics)
                        latest_postfix = _progress_postfix(
                            step=step,
                            lr=lr_value,
                            loss_value=loss_value,
                            tokens_per_sec=metrics["perf/tokens_per_sec"],
                            routing_metrics=routing_metrics,
                        )
                        if tracker is not None:
                            tracker.log_metrics(metrics, step)

                    if progress is not None:
                        if latest_postfix is not None:
                            progress.set_postfix(latest_postfix, refresh=False)
                        progress.update(1)

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

        # Save the final checkpoint if it wasn't saved perfectly on the modulo boundary
        if ctx.is_main and int(cfg.trainer.checkpoint_every) > 0 and step % int(cfg.trainer.checkpoint_every) != 0:
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
