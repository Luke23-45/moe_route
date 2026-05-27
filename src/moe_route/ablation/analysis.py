"""Cross-run ablation result analysis and comparison.

Provides utilities to load ablation results from the persistence layer,
compute comparative statistics, rank variants, and generate publication-
quality summary tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_study_results(study_dir: Path) -> dict[str, Any]:
    """Load the study summary from a persistence directory.

    Parameters
    ----------
    study_dir : Path
        Path to the study directory (e.g.
        ``artifacts/results/phase_b_ablation_reflected``).

    Returns
    -------
    dict
        The study summary, or empty dict if not found.
    """
    summary_path = study_dir / "study_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def rank_by_metric(
    study_results: dict[str, Any],
    metric_key: str = "final_train_loss",
    *,
    lower_is_better: bool = True,
) -> list[dict[str, Any]]:
    """Rank experiments by a given metric.

    Parameters
    ----------
    study_results : dict
        Output of ``load_study_results()`` or
        ``ResultStore.generate_study_summary()``.
    metric_key : str
        The metric key to rank by (looked up in the ``mean`` dict
        of each experiment's aggregation).
    lower_is_better : bool
        If True, lower values are better (e.g. loss).

    Returns
    -------
    list[dict]
        Sorted list of ``{"experiment_id", "mean", "std", "n_seeds", "rank"}``
        dicts, ordered from best to worst.
    """
    experiments = study_results.get("experiments", {})
    entries: list[dict[str, Any]] = []

    for exp_id, agg in experiments.items():
        means = agg.get("mean", {})
        stds = agg.get("std", {})
        n_seeds = agg.get("n_seeds", 0)

        val = means.get(metric_key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue

        entries.append({
            "experiment_id": exp_id,
            "mean": val,
            "std": stds.get(metric_key, 0.0),
            "n_seeds": n_seeds,
        })

    entries.sort(key=lambda e: e["mean"], reverse=not lower_is_better)

    for i, entry in enumerate(entries, 1):
        entry["rank"] = i

    return entries


def compute_relative_improvement(
    ranked: list[dict[str, Any]],
    baseline_id: str,
) -> list[dict[str, Any]]:
    """Add relative improvement over a baseline to ranked results.

    Parameters
    ----------
    ranked : list[dict]
        Output of ``rank_by_metric()``.
    baseline_id : str
        The ``experiment_id`` of the baseline to compare against.

    Returns
    -------
    list[dict]
        The ranked list with added ``"delta"`` and ``"delta_pct"`` fields.
        ``delta`` is absolute (method - baseline), ``delta_pct`` is
        ``100 * delta / baseline``.
    """
    baseline_val = None
    for entry in ranked:
        if entry["experiment_id"] == baseline_id:
            baseline_val = entry["mean"]
            break

    if baseline_val is None or baseline_val == 0:
        return ranked

    for entry in ranked:
        entry["delta"] = entry["mean"] - baseline_val
        entry["delta_pct"] = 100.0 * entry["delta"] / abs(baseline_val)

    return ranked


def format_ablation_table(
    ranked: list[dict[str, Any]],
    *,
    metric_name: str = "Loss",
    show_delta: bool = True,
) -> str:
    """Format ranked ablation results as an ASCII table.

    Parameters
    ----------
    ranked : list[dict]
        Output of ``rank_by_metric()`` or ``compute_relative_improvement()``.
    metric_name : str
        Label for the metric column.
    show_delta : bool
        If True and ``delta_pct`` is present, show a delta column.

    Returns
    -------
    str
        Formatted ASCII table string.
    """
    has_delta = show_delta and any("delta_pct" in e for e in ranked)

    lines: list[str] = []
    header = f"{'Rank':>4}  {'Experiment':<45}  {'Seeds':>5}  {metric_name + ' (mean±std)':>22}"
    if has_delta:
        header += f"  {'Δ%':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for entry in ranked:
        mean = entry["mean"]
        std = entry["std"]
        n = entry["n_seeds"]
        rank = entry["rank"]
        exp_id = entry["experiment_id"]

        if n > 1:
            metric_str = f"{mean:.4f} ± {std:.4f}"
        else:
            metric_str = f"{mean:.4f}"

        line = f"{rank:>4}  {exp_id:<45}  {n:>5}  {metric_str:>22}"

        if has_delta and "delta_pct" in entry:
            sign = "+" if entry["delta_pct"] >= 0 else ""
            line += f"  {sign}{entry['delta_pct']:.2f}%"

        lines.append(line)

    return "\n".join(lines)


def identify_best_config(
    study_dir: Path,
    *,
    metric_key: str = "final_train_loss",
    exclude_references: bool = True,
) -> dict[str, Any] | None:
    """Identify the best ablation configuration from study results.

    Parameters
    ----------
    study_dir : Path
        Path to the study directory.
    metric_key : str
        Metric to optimize.
    exclude_references : bool
        If True, exclude reference baselines (top2, deepseek_lfb) from
        the ranking.

    Returns
    -------
    dict | None
        The best entry from ``rank_by_metric()``, or None if no results.
    """
    results = load_study_results(study_dir)
    if not results:
        return None

    # Filter out reference baselines if requested
    if exclude_references:
        experiments = results.get("experiments", {})
        filtered = {
            k: v for k, v in experiments.items()
            if not k.startswith("top2") and not k.startswith("deepseek_lfb")
        }
        results = dict(results)
        results["experiments"] = filtered

    ranked = rank_by_metric(results, metric_key, lower_is_better=True)
    return ranked[0] if ranked else None
