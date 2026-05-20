from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from moe_route.data.pipeline import build_dataloader
from moe_route.evaluation.ppl import build_eval_data_cfg, load_cfg_from_checkpoint
from moe_route.models.transformer import DecoderOnlyLM, build_model_cfg
from moe_route.tokenization.tokenizers import build_tokenizer
from moe_route.training.checkpoint import load_checkpoint


def get_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _load_validation_rows(validation_metrics_csv: Path) -> list[dict[str, str]]:
    if not validation_metrics_csv.exists():
        return []
    with validation_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_checkpoints(
    checkpoint_dir: str | Path,
    validation_metrics_csv: str | Path | None = None,
    *,
    eval_split: str | None = None,
) -> dict[str, Path]:
    checkpoint_root = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_root.glob("step_*.pt"), key=get_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_root}")

    latest = checkpoints[-1]
    selected = {"latest": latest}

    if validation_metrics_csv is None:
        selected["best"] = latest
        return selected

    rows = _load_validation_rows(Path(validation_metrics_csv))
    if not rows:
        selected["best"] = latest
        return selected

    candidates: list[tuple[float, int]] = []
    for row in rows:
        if "eval/loss" not in row or "step" not in row:
            continue
        if eval_split is not None and str(row.get("eval_split", "")) != eval_split:
            continue
        try:
            candidates.append((float(row["eval/loss"]), int(row["step"])))
        except ValueError:
            continue

    if not candidates:
        selected["best"] = latest
        return selected

    _, best_step = min(candidates, key=lambda item: (item[0], -item[1]))
    best = checkpoint_root / f"step_{best_step}.pt"
    selected["best"] = best if best.exists() else latest
    return selected


@torch.no_grad()
def evaluate_routing_stress(
    model,
    dataloader,
    max_batches: int,
    device: torch.device,
    *,
    ignore_index: int = -100,
    precision: str = "fp32",
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_aux = 0.0
    aux_batches = 0
    router_stats: dict[int, dict[str, list[float] | float]] = {}

    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits, _, parts = model(input_ids, labels)

        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )
        if ignore_index >= 0:
            token_count = int((labels != ignore_index).sum().item())
        else:
            token_count = int(labels.numel())
        total_nll += float(nll.detach().cpu())
        total_tokens += token_count
        if isinstance(parts, dict) and "aux_loss" in parts:
            total_aux += float(parts["aux_loss"].detach().cpu())
            aux_batches += 1

        diagnostics = model.routing_diagnostics() if hasattr(model, "routing_diagnostics") else []
        for i, diag in enumerate(diagnostics):
            top_k = max(int(diag.extra.get("indices", torch.empty(0, 1)).shape[-1]), 1)
            requested_assignments = max(int(diag.raw_load.sum().item()), 1)
            capacity_sum = max(float(diag.capacity.sum().item()), 1.0)
            stats = router_stats.setdefault(
                i,
                {
                    "token_weight": 0.0,
                    "assignment_weight": 0.0,
                    "capacity_weight": 0.0,
                    "drop_rate_weighted": 0.0,
                    "dropped_assignments_weighted": 0.0,
                    "entropy_weighted": 0.0,
                    "matched_compute_weighted": 0.0,
                    "capacity_utilization_weighted": 0.0,
                    "load_cv_weighted": 0.0,
                    "load_gini_weighted": 0.0,
                    "overflow_total": 0.0,
                    "pressure_mean_weighted": 0.0,
                },
            )

            raw_load = diag.raw_load.float()
            mean_raw_load = raw_load.mean()
            if float(mean_raw_load) > 0:
                load_cv = float((raw_load.std() / (mean_raw_load + 1e-8)).detach().cpu())
                diff = torch.abs(raw_load.unsqueeze(0) - raw_load.unsqueeze(1))
                load_gini = float(
                    (diff.sum() / (2.0 * raw_load.numel() * raw_load.sum() + 1e-8)).detach().cpu()
                )
            else:
                load_cv = 0.0
                load_gini = 0.0

            stats["token_weight"] += token_count
            stats["assignment_weight"] += requested_assignments
            stats["capacity_weight"] += capacity_sum
            stats["drop_rate_weighted"] += float(diag.dropped.detach().cpu()) * token_count
            stats["dropped_assignments_weighted"] += (
                float(diag.dropped_assignments.detach().cpu()) * requested_assignments
            )
            stats["entropy_weighted"] += float(diag.entropy.detach().cpu()) * token_count
            stats["matched_compute_weighted"] += (
                float(diag.matched_compute_fraction.detach().cpu()) * requested_assignments
            )
            stats["capacity_utilization_weighted"] += (
                float(diag.capacity_utilization.detach().cpu()) * capacity_sum
            )
            stats["load_cv_weighted"] += load_cv * token_count
            stats["load_gini_weighted"] += load_gini * token_count
            stats["overflow_total"] += float(diag.overflow.float().sum().detach().cpu())
            if diag.pressure is not None:
                stats["pressure_mean_weighted"] += (
                    float(diag.pressure.float().mean().detach().cpu()) * token_count
                )

    mean_loss = total_nll / max(total_tokens, 1)
    metrics = {
        "eval/loss": mean_loss,
        "eval/lm_loss": mean_loss,
        "eval/ppl": math.exp(min(mean_loss, 30.0)),
        "eval/bpb": mean_loss / math.log(2),
        "eval/tokens": total_tokens,
        "eval/batches": max_batches if max_batches > 0 else 0,
    }
    if aux_batches:
        metrics["eval/aux_loss"] = total_aux / aux_batches

    for router_idx, stats in router_stats.items():
        prefix = f"router/{router_idx}"
        token_weight = max(float(stats["token_weight"]), 1.0)
        assignment_weight = max(float(stats["assignment_weight"]), 1.0)
        capacity_weight = max(float(stats["capacity_weight"]), 1.0)
        metrics[f"{prefix}/drop_rate"] = float(stats["drop_rate_weighted"]) / token_weight
        metrics[f"{prefix}/dropped_assignments"] = (
            float(stats["dropped_assignments_weighted"]) / assignment_weight
        )
        metrics[f"{prefix}/entropy"] = float(stats["entropy_weighted"]) / token_weight
        metrics[f"{prefix}/matched_compute_fraction"] = (
            float(stats["matched_compute_weighted"]) / assignment_weight
        )
        metrics[f"{prefix}/capacity_utilization"] = (
            float(stats["capacity_utilization_weighted"]) / capacity_weight
        )
        metrics[f"{prefix}/load_cv"] = float(stats["load_cv_weighted"]) / token_weight
        metrics[f"{prefix}/load_gini"] = float(stats["load_gini_weighted"]) / token_weight
        metrics[f"{prefix}/overflow"] = float(stats["overflow_total"])
        if float(stats["pressure_mean_weighted"]) > 0.0:
            metrics[f"{prefix}/pressure_mean"] = float(stats["pressure_mean_weighted"]) / token_weight

    return metrics


def evaluate_checkpoint_routing_stress(
    checkpoint: str | Path,
    *,
    max_batches: int | None = None,
) -> dict[str, float | str]:
    checkpoint_path = Path(checkpoint)
    cfg = load_cfg_from_checkpoint(checkpoint_path)
    eval_cfg = build_eval_data_cfg(cfg)
    tokenizer = build_tokenizer(cfg.tokenizer)
    dataloader = build_dataloader(eval_cfg, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderOnlyLM(build_model_cfg(cfg)).to(device)
    load_checkpoint(checkpoint_path, model)
    resolved_max_batches = int(cfg.eval.max_batches if max_batches is None else max_batches)
    precision = str(cfg.trainer.get("precision", "fp32"))
    pad_token_id = tokenizer.pad_token_id
    metrics = evaluate_routing_stress(
        model, dataloader, resolved_max_batches, device,
        ignore_index=pad_token_id,
        precision=precision,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["eval_split"] = str(eval_cfg.get("split", ""))
    metrics["prepared_split"] = str(eval_cfg.get("prepared_split", "train"))
    return metrics


def write_routing_stress_report(
    output_json: str | Path,
    payload: dict[str, object],
    *,
    output_md: str | Path | None = None,
) -> None:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if output_md is None:
        return

    md_path = Path(output_md)
    lines = [
        "# Routing Stress Evaluation",
        "",
        f"- run: `{payload.get('run_name', '')}`",
        f"- eval_split: `{payload.get('eval_split', '')}`",
        f"- prepared_split: `{payload.get('prepared_split', '')}`",
        "",
        "| Checkpoint | Eval Loss | Eval PPL | Drop Rate | Dropped Assignments | Matched Compute | Load CV | Load Gini | Overflow |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in payload.get("checkpoints", {}).items():
        row = metrics
        lines.append(
            "| {label} | {loss:.6f} | {ppl:.6f} | {drop:.6f} | {dropped:.6f} | {compute:.6f} | {cv:.6f} | {gini:.6f} | {overflow:.2f} |".format(
                label=label,
                loss=float(row.get("eval/loss", float("nan"))),
                ppl=float(row.get("eval/ppl", float("nan"))),
                drop=float(row.get("router/0/drop_rate", 0.0)),
                dropped=float(row.get("router/0/dropped_assignments", 0.0)),
                compute=float(row.get("router/0/matched_compute_fraction", 0.0)),
                cv=float(row.get("router/0/load_cv", 0.0)),
                gini=float(row.get("router/0/load_gini", 0.0)),
                overflow=float(row.get("router/0/overflow", 0.0)),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
