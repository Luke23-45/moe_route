"""Research-grade perplexity evaluation for language models.

Implements:
  - Pad / special-token masking (ignore_index)
  - Per-batch NLL tracking with confidence intervals (ppl_stderr)
  - AMP support matching training precision
  - Bits-per-byte (BPB) tokenizer-invariant metric
  - Robust numerical handling with warnings
  - Domain-stratified PPL (dialogue / punctuation / normal)
  - Per-position loss curve (early / middle / late context)
  - Tail analysis (worst-case 90th/95th percentile per-sample loss)
  - Top-k token prediction accuracy (top1_acc, top5_acc)
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from moe_route.data.pipeline import build_dataloader
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

# Maximum mean NLL before we cap exp() to avoid overflow.
_MAX_LOSS_FOR_EXP = 30.0


# ────────────────────────────────────────────────────────────────────
# Domain classification helpers (tokenizer-aware)
# ────────────────────────────────────────────────────────────────────

def _resolve_domain_token_ids(tokenizer) -> tuple[list[int], list[int]] | None:
    """Try to resolve quote and punctuation token IDs from the tokenizer.

    Returns (quote_ids, punct_ids) or None if the tokenizer doesn't support
    single-character encoding.
    """
    try:
        from moe_route.tokenization.tokenizers import resolve_char_token_ids
        char_ids = resolve_char_token_ids(tokenizer, '".,?!')
        quote_ids = [char_ids['"']]
        punct_ids = [char_ids[c] for c in '.,?!']
        return quote_ids, punct_ids
    except (ValueError, ImportError):
        return None


def _classify_domains(
    labels: torch.Tensor,
    quote_ids: list[int],
    punct_ids: list[int],
    dialogue_threshold: int = 4,
    punct_threshold: int = 11,
) -> torch.Tensor:
    """Classify each sample in a batch into domain categories.

    Returns:
        [B] tensor with values: 0=normal, 1=dialogue, 2=punctuation
    """
    B = labels.shape[0]

    # Count quotes
    quote_mask = torch.zeros_like(labels, dtype=torch.bool)
    for qid in quote_ids:
        quote_mask |= labels == qid
    quote_counts = quote_mask.sum(dim=-1)

    # Count punctuation
    punct_mask = torch.zeros_like(labels, dtype=torch.bool)
    for pid in punct_ids:
        punct_mask |= labels == pid
    punct_counts = punct_mask.sum(dim=-1)

    is_dialogue = quote_counts >= dialogue_threshold
    is_punct = (punct_counts >= punct_threshold) & (~is_dialogue)

    domain_labels = torch.zeros(B, dtype=torch.long, device=labels.device)
    domain_labels[is_dialogue] = 1
    domain_labels[is_punct] = 2
    return domain_labels


# ────────────────────────────────────────────────────────────────────
# Core evaluation
# ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model_perplexity(
    model: torch.nn.Module,
    dataloader,
    max_batches: int,
    device: torch.device,
    *,
    ignore_index: int = -100,
    precision: str = "fp32",
    tokenizer=None,
) -> dict[str, float]:
    """Evaluate perplexity with research-grade robustness and diagnostic depth.

    Computes aggregate metrics plus domain-stratified PPL, per-position loss
    curves, tail analysis, and top-k accuracy — all designed to expose
    robustness differences between routing strategies.

    Args:
        model: Language model in eval mode.
        dataloader: Yields (input_ids, labels) batches.
        max_batches: Maximum batches to evaluate (0 = all).
        device: Target device.
        ignore_index: Token ID to exclude from loss computation.
        precision: AMP precision ("bf16", "fp16", or "fp32").
        tokenizer: Optional tokenizer for domain-stratified evaluation.

    Returns:
        Dictionary with comprehensive evaluation metrics.
    """
    model.eval()

    # AMP settings
    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # ── Aggregate accumulators ──
    total_nll = 0.0
    total_tokens = 0
    total_aux = 0.0
    aux_batches = 0

    # ── Per-batch loss for confidence intervals ──
    batch_losses: list[float] = []
    batch_token_counts: list[int] = []

    # ── Per-sample loss for tail analysis ──
    sample_losses: list[float] = []

    # ── Top-k accuracy accumulators ──
    top1_correct = 0
    top5_correct = 0
    topk_total = 0

    # ── Per-position loss accumulators ──
    # Track NLL at each position for positional analysis
    seq_len_observed = 0
    position_nll: torch.Tensor | None = None  # [S] accumulator
    position_counts: torch.Tensor | None = None  # [S] count

    # ── Domain-stratified accumulators ──
    # 0=normal, 1=dialogue, 2=punctuation
    domain_names = ["normal", "dialogue", "punctuation"]
    domain_nll = {i: 0.0 for i in range(3)}
    domain_tokens = {i: 0 for i in range(3)}
    domain_ids = None
    if tokenizer is not None:
        domain_ids = _resolve_domain_token_ids(tokenizer)

    num_batches = 0
    for idx, (input_ids, labels) in enumerate(dataloader):
        if 0 < max_batches <= idx:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B, S = labels.shape

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits, _, parts = model(input_ids, labels)

        # ── Per-token NLL (no reduction) ──
        per_token_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(B, S)  # [B, S]

        # Build valid mask
        if ignore_index >= 0:
            valid_mask = labels != ignore_index  # [B, S]
        else:
            valid_mask = torch.ones_like(labels, dtype=torch.bool)

        # Zero out NLL for masked positions
        per_token_nll = per_token_nll * valid_mask.float()

        token_count = int(valid_mask.sum().item())
        if token_count == 0:
            continue

        batch_nll = float(per_token_nll.sum().item())

        if math.isnan(batch_nll) or math.isinf(batch_nll):
            warnings.warn(
                f"[eval] NaN/Inf NLL at batch {idx} — skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # ── Aggregate ──
        total_nll += batch_nll
        total_tokens += token_count
        batch_losses.append(batch_nll / token_count)
        batch_token_counts.append(token_count)
        num_batches += 1

        if isinstance(parts, dict) and "aux_loss" in parts:
            total_aux += float(parts["aux_loss"].detach().cpu())
            aux_batches += 1

        # ── Per-sample loss for tail analysis ──
        sample_valid_counts = valid_mask.sum(dim=-1).float().clamp_min(1)  # [B]
        sample_nlls = per_token_nll.sum(dim=-1) / sample_valid_counts  # [B]
        sample_losses.extend(sample_nlls.cpu().tolist())

        # ── Top-k accuracy ──
        flat_logits = logits.reshape(-1, logits.size(-1))  # [B*S, V]
        flat_labels = labels.reshape(-1)  # [B*S]
        if ignore_index >= 0:
            flat_valid = flat_labels != ignore_index
        else:
            flat_valid = torch.ones(flat_labels.shape[0], dtype=torch.bool, device=device)

        if flat_valid.any():
            valid_logits = flat_logits[flat_valid]
            valid_labels = flat_labels[flat_valid]
            n_valid = valid_labels.shape[0]

            # Top-1
            top1_preds = valid_logits.argmax(dim=-1)
            top1_correct += int((top1_preds == valid_labels).sum().item())

            # Top-5
            k = min(5, valid_logits.size(-1))
            top5_preds = valid_logits.topk(k, dim=-1).indices  # [n, 5]
            top5_hits = (top5_preds == valid_labels.unsqueeze(-1)).any(dim=-1)
            top5_correct += int(top5_hits.sum().item())
            topk_total += n_valid

        # ── Per-position loss accumulation ──
        if position_nll is None:
            position_nll = torch.zeros(S, dtype=torch.float64)
            position_counts = torch.zeros(S, dtype=torch.float64)
            seq_len_observed = S

        if S == seq_len_observed:
            pos_nll_batch = per_token_nll.sum(dim=0).cpu().double()  # [S]
            pos_count_batch = valid_mask.sum(dim=0).cpu().double()  # [S]
            position_nll += pos_nll_batch
            position_counts += pos_count_batch

        # ── Domain-stratified NLL ──
        if domain_ids is not None:
            quote_ids, punct_ids = domain_ids
            domains = _classify_domains(labels, quote_ids, punct_ids)  # [B]
            for d in range(3):
                mask_d = domains == d  # [B]
                if mask_d.any():
                    d_nll = per_token_nll[mask_d].sum().item()
                    d_tok = valid_mask[mask_d].sum().item()
                    domain_nll[d] += d_nll
                    domain_tokens[d] += int(d_tok)

    # ────────────────────────────────────────────────────────────────
    # Compute final metrics
    # ────────────────────────────────────────────────────────────────

    if total_tokens == 0:
        warnings.warn(
            "[eval] No valid tokens evaluated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "eval/loss": float("nan"),
            "eval/lm_loss": float("nan"),
            "eval/ppl": float("nan"),
            "eval/ppl_stderr": float("nan"),
            "eval/bpb": float("nan"),
            "eval/tokens": 0,
            "eval/batches": 0,
        }

    mean_loss = total_nll / total_tokens

    if mean_loss > _MAX_LOSS_FOR_EXP:
        warnings.warn(
            f"[eval] Mean NLL = {mean_loss:.4f} exceeds {_MAX_LOSS_FOR_EXP}. "
            f"Model may not have learned or checkpoint is corrupted.",
            RuntimeWarning,
            stacklevel=2,
        )

    ppl = math.exp(min(mean_loss, _MAX_LOSS_FOR_EXP))
    bpb = mean_loss / math.log(2)

    # ── Confidence intervals ──
    ppl_stderr = 0.0
    if len(batch_losses) > 1:
        total_weight = sum(batch_token_counts)
        weighted_mean = sum(
            bl * bc for bl, bc in zip(batch_losses, batch_token_counts)
        ) / total_weight
        weighted_var = sum(
            bc * (bl - weighted_mean) ** 2
            for bl, bc in zip(batch_losses, batch_token_counts)
        ) / total_weight
        n_eff = len(batch_losses)
        loss_stderr = math.sqrt(weighted_var / n_eff) if n_eff > 0 else 0.0
        ppl_stderr = ppl * loss_stderr

    metrics: dict[str, float] = {
        "eval/loss": mean_loss,
        "eval/lm_loss": mean_loss,
        "eval/ppl": ppl,
        "eval/ppl_stderr": ppl_stderr,
        "eval/bpb": bpb,
        "eval/tokens": total_tokens,
        "eval/batches": num_batches,
    }
    if aux_batches:
        metrics["eval/aux_loss"] = total_aux / aux_batches

    # ── Top-k accuracy ──
    if topk_total > 0:
        metrics["eval/top1_acc"] = top1_correct / topk_total
        metrics["eval/top5_acc"] = top5_correct / topk_total

    # ── Tail analysis (per-sample loss distribution) ──
    if sample_losses:
        sorted_losses = sorted(sample_losses)
        n = len(sorted_losses)
        metrics["eval/median_loss"] = sorted_losses[n // 2]
        metrics["eval/p90_loss"] = sorted_losses[int(n * 0.90)]
        metrics["eval/p95_loss"] = sorted_losses[int(min(n * 0.95, n - 1))]
        metrics["eval/p99_loss"] = sorted_losses[int(min(n * 0.99, n - 1))]
        metrics["eval/max_sample_loss"] = sorted_losses[-1]
        metrics["eval/min_sample_loss"] = sorted_losses[0]
        # Tail-to-median ratio: how much worse are the worst cases?
        median = sorted_losses[n // 2]
        if median > 0:
            metrics["eval/p95_to_median_ratio"] = sorted_losses[int(min(n * 0.95, n - 1))] / median

    # ── Per-position loss curve (quintile summary) ──
    if position_nll is not None and position_counts is not None:
        safe_counts = position_counts.clamp_min(1)
        pos_mean_loss = (position_nll / safe_counts).tolist()
        S = len(pos_mean_loss)
        if S >= 10:
            # Report loss in 5 position bands
            band_size = S // 5
            for band_idx, band_name in enumerate(["early", "mid_early", "middle", "mid_late", "late"]):
                start = band_idx * band_size
                end = start + band_size if band_idx < 4 else S
                band_vals = pos_mean_loss[start:end]
                metrics[f"eval/pos_{band_name}_loss"] = sum(band_vals) / len(band_vals)

            # Early vs late ratio: shows context utilization effectiveness
            early_loss = sum(pos_mean_loss[:band_size]) / band_size
            late_loss = sum(pos_mean_loss[-band_size:]) / band_size
            if late_loss > 0:
                metrics["eval/early_to_late_ratio"] = early_loss / late_loss

    # ── Domain-stratified PPL ──
    if domain_ids is not None:
        for d in range(3):
            name = domain_names[d]
            if domain_tokens[d] > 0:
                d_loss = domain_nll[d] / domain_tokens[d]
                d_ppl = math.exp(min(d_loss, _MAX_LOSS_FOR_EXP))
                metrics[f"eval/domain_{name}_loss"] = d_loss
                metrics[f"eval/domain_{name}_ppl"] = d_ppl
                metrics[f"eval/domain_{name}_tokens"] = domain_tokens[d]

        # Cross-domain PPL spread: how different are the domain PPLs?
        domain_ppls = []
        for d in range(3):
            if domain_tokens[d] > 0:
                domain_ppls.append(math.exp(min(domain_nll[d] / domain_tokens[d], _MAX_LOSS_FOR_EXP)))
        if len(domain_ppls) >= 2:
            metrics["eval/domain_ppl_spread"] = max(domain_ppls) / max(min(domain_ppls), 1e-8)
            metrics["eval/domain_ppl_std"] = float(
                torch.tensor(domain_ppls).std().item()
            )

    return metrics


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _has_prepared_holdout_shard(eval_data_cfg, tokenizer) -> bool:
    """Check if a prepared holdout validation shard exists on disk with a valid manifest.

    This mirrors how the training pipeline detects and reuses prepared data.
    Returns True only when the holdout validation binary and its manifest both
    exist and the manifest fingerprint matches the current config+tokenizer.
    """
    if float(eval_data_cfg.get("holdout_fraction", 0.0)) <= 0.0:
        return False
    if str(eval_data_cfg.get("split", "")) != "train":
        return False
    try:
        from moe_route.data.manifest import (
            dataset_fingerprint,
            prepared_dataset_paths,
            read_manifest,
        )

        val_probe = OmegaConf.create(OmegaConf.to_container(eval_data_cfg, resolve=True))
        val_probe.prepared_split = "validation"
        samples_path, manifest_path = prepared_dataset_paths(val_probe, tokenizer)
        if not samples_path.exists() or not manifest_path.exists():
            return False
        manifest = read_manifest(manifest_path)
        return manifest.get("fingerprint") == dataset_fingerprint(val_probe, tokenizer)
    except Exception:
        return False


def build_eval_data_cfg(cfg, tokenizer=None):
    """Build a data config suitable for evaluation.

    Tries to find a validation/test split automatically:
    1. Explicit override via cfg.eval.data_split
    2. Pre-prepared holdout validation shard on disk (from training data preparation)
    3. Probing HuggingFace dataset for validation/valid/test splits
    4. Falling back to holdout fraction from training data
    """
    eval_data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    eval_data_cfg.prepared_split = str(eval_data_cfg.get("prepared_split", "train"))

    # Disable shuffling for deterministic evaluation
    eval_data_cfg.shuffle = False

    split_override = cfg.eval.get("data_split", None)
    if split_override is not None:
        eval_data_cfg.split = str(split_override)
        eval_data_cfg.prepared_split = "validation"
        return eval_data_cfg

    # Intelligent detection: prefer an already-prepared holdout validation shard
    # over downloading a native HuggingFace split.  This mirrors how the training
    # pipeline automatically detects and reuses prepared data.
    if tokenizer is not None and _has_prepared_holdout_shard(eval_data_cfg, tokenizer):
        logger.info(
            "[eval] Found prepared holdout validation shard on disk — using it "
            "instead of probing for a native HuggingFace split."
        )
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
    """Full evaluation pipeline: load model, data, and compute perplexity.

    Uses pad_token_id from the tokenizer as ignore_index, and matches
    training precision for consistent results. Passes the tokenizer for
    domain-stratified evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg.tokenizer)
    dataloader = build_dataloader(build_eval_data_cfg(cfg, tokenizer=tokenizer), tokenizer)
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(device)
    if checkpoint is not None:
        load_checkpoint(checkpoint, model)

    max_batches = int(cfg.eval.max_batches)
    precision = str(cfg.trainer.get("precision", "fp32"))
    pad_token_id = tokenizer.pad_token_id

    logger.info(
        "[eval] Evaluating perplexity: max_batches=%d, precision=%s, "
        "ignore_index=%d (pad_token_id), device=%s",
        max_batches, precision, pad_token_id, device,
    )

    return evaluate_model_perplexity(
        model, dataloader, max_batches, device,
        ignore_index=pad_token_id,
        precision=precision,
        tokenizer=tokenizer,
    )


def load_cfg_from_checkpoint(checkpoint: str | Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return OmegaConf.create(payload["config"])
