"""Hierarchical result persistence and aggregation.

Provides ``ResultStore`` — the single entry point for storing, loading,
and aggregating experiment results across seeds and methods.

Directory layout::

    {result_root}/
    └── {study_name}/
        ├── {experiment_id}/
        │   ├── seed_{seed}/
        │   │   ├── manifest.json         # Complete run manifest
        │   │   ├── metrics.jsonl         # Copied training metrics
        │   │   ├── checkpoint_path.txt   # Pointer to checkpoint
        │   │   └── summary.json          # Final summary metrics
        │   └── aggregate.json            # Cross-seed statistics
        └── study_summary.json            # Full study comparison report

The layout is common across *all* experiment types (baseline, ablation,
fineweb) so that downstream analysis can operate uniformly on any study.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from studies.utils.manifest import RunManifest


class ResultStore:
    """Hierarchical result persistence for a single study.

    Parameters
    ----------
    result_root : Path | str
        Root directory for all results (typically ``artifacts/results``).
    study_name : str
        Name of the study (e.g. ``"phase_a_tinystories_baseline"``).
    """

    def __init__(self, result_root: Path | str, study_name: str) -> None:
        self.result_root = Path(result_root)
        self.study_name = study_name
        self.study_dir = self.result_root / study_name
        self.study_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------

    def run_dir(self, experiment_id: str, seed: int) -> Path:
        """Return the directory for a specific (experiment_id, seed) run."""
        d = self.study_dir / experiment_id / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def experiment_dir(self, experiment_id: str) -> Path:
        """Return the directory for an experiment (across seeds)."""
        d = self.study_dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -----------------------------------------------------------------
    # Per-run persistence
    # -----------------------------------------------------------------

    def save_manifest(self, manifest: RunManifest, experiment_id: str, seed: int) -> Path:
        """Save a run manifest to the appropriate directory."""
        path = self.run_dir(experiment_id, seed) / "manifest.json"
        manifest.save(path)
        return path

    def save_metrics(
        self,
        experiment_id: str,
        seed: int,
        source_metrics_path: Path | str,
    ) -> Path | None:
        """Copy training metrics.jsonl into the persistence directory.

        Returns the destination path, or ``None`` if the source doesn't exist.
        """
        src = Path(source_metrics_path)
        if not src.exists():
            return None
        dst = self.run_dir(experiment_id, seed) / "metrics.jsonl"
        shutil.copy2(src, dst)
        return dst

    def save_checkpoint_pointer(
        self,
        experiment_id: str,
        seed: int,
        checkpoint_path: str | Path | None,
    ) -> Path | None:
        """Write a text file pointing to the checkpoint location."""
        if checkpoint_path is None:
            return None
        dst = self.run_dir(experiment_id, seed) / "checkpoint_path.txt"
        dst.write_text(str(checkpoint_path), encoding="utf-8")
        return dst

    def save_summary(
        self,
        experiment_id: str,
        seed: int,
        summary: dict[str, Any],
    ) -> Path:
        """Save per-run summary metrics (e.g. final loss, throughput)."""
        path = self.run_dir(experiment_id, seed) / "summary.json"
        path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    # -----------------------------------------------------------------
    # Aggregation across seeds
    # -----------------------------------------------------------------

    def load_summaries(self, experiment_id: str) -> dict[int, dict[str, Any]]:
        """Load all per-seed summaries for an experiment.

        Returns a mapping from seed → summary dict.
        """
        exp_dir = self.study_dir / experiment_id
        if not exp_dir.exists():
            return {}

        summaries: dict[int, dict[str, Any]] = {}
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            summary_path = seed_dir / "summary.json"
            if summary_path.exists():
                seed_val = int(seed_dir.name.removeprefix("seed_"))
                summaries[seed_val] = json.loads(summary_path.read_text(encoding="utf-8"))
        return summaries

    def aggregate_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Compute cross-seed statistics for an experiment.

        Calculates mean and standard deviation for all numeric fields
        found in the per-seed summaries.  The result is saved to
        ``aggregate.json`` in the experiment directory.

        Returns
        -------
        dict
            The aggregation result including ``n_seeds``, ``seeds``,
            ``mean``, ``std``, and ``per_seed`` data.
        """
        summaries = self.load_summaries(experiment_id)
        if not summaries:
            return {}

        seeds = sorted(summaries.keys())
        n = len(seeds)

        # Collect all numeric keys across summaries
        numeric_keys: set[str] = set()
        for s in summaries.values():
            for k, v in s.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    numeric_keys.add(k)

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for key in sorted(numeric_keys):
            values = [summaries[seed][key] for seed in seeds if key in summaries[seed]]
            if not values:
                continue
            mean = sum(values) / len(values)
            means[key] = mean
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                stds[key] = math.sqrt(variance)
            else:
                stds[key] = 0.0

        aggregate = {
            "experiment_id": experiment_id,
            "n_seeds": n,
            "seeds": seeds,
            "mean": means,
            "std": stds,
            "per_seed": {str(seed): summaries[seed] for seed in seeds},
        }

        path = self.experiment_dir(experiment_id) / "aggregate.json"
        path.write_text(
            json.dumps(aggregate, indent=2, default=str),
            encoding="utf-8",
        )
        return aggregate

    # -----------------------------------------------------------------
    # Study-level summary
    # -----------------------------------------------------------------

    def generate_study_summary(self) -> dict[str, Any]:
        """Generate a full study comparison report across all experiments.

        Iterates over all experiment directories, aggregates each one,
        and compiles a study-level summary for cross-method comparison.

        Returns
        -------
        dict
            Study summary with per-experiment aggregations and
            a comparison table of key metrics.
        """
        experiments: dict[str, Any] = {}
        if not self.study_dir.exists():
            return {}

        for exp_dir in sorted(self.study_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            # Check if this directory has seed subdirectories
            seed_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
            if not seed_dirs:
                continue
            agg = self.aggregate_experiment(exp_dir.name)
            if agg:
                experiments[exp_dir.name] = agg

        summary = {
            "study_name": self.study_name,
            "n_experiments": len(experiments),
            "experiments": experiments,
        }

        path = self.study_dir / "study_summary.json"
        path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        return summary

    # -----------------------------------------------------------------
    # Result collection from training artifacts
    # -----------------------------------------------------------------

    def collect_run_results(
        self,
        experiment_id: str,
        seed: int,
        manifest: RunManifest,
        run_artifact_dir: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> None:
        """Collect all results from a completed training run.

        This is the primary post-run entry point.  It:
          1. Saves the finalized manifest
          2. Copies metrics.jsonl from the training artifact directory
          3. Records the checkpoint path
          4. Extracts summary metrics from the metrics log
          5. Saves the per-run summary
        """
        self.save_manifest(manifest, experiment_id, seed)

        if run_artifact_dir is not None:
            art_dir = Path(run_artifact_dir)
            metrics_src = art_dir / "metrics.jsonl"
            self.save_metrics(experiment_id, seed, metrics_src)

        self.save_checkpoint_pointer(experiment_id, seed, checkpoint_path)

        # Extract summary from manifest
        summary: dict[str, Any] = {
            "seed": seed,
            "experiment_id": experiment_id,
            "run_name": manifest.run_name,
            "status": manifest.status,
            "wall_clock_seconds": manifest.wall_clock_seconds,
            "total_tokens": manifest.total_tokens,
            "tokens_per_second": manifest.tokens_per_second,
        }
        if manifest.final_train_loss is not None:
            summary["final_train_loss"] = manifest.final_train_loss
        if manifest.final_val_loss is not None:
            summary["final_val_loss"] = manifest.final_val_loss
        if manifest.checkpoint_path is not None:
            summary["checkpoint_path"] = manifest.checkpoint_path

        # Also try to extract final metrics from metrics.jsonl
        final_metrics = self._extract_final_metrics(experiment_id, seed)
        if final_metrics:
            summary["final_metrics"] = final_metrics

        self.save_summary(experiment_id, seed, summary)

    def _extract_final_metrics(
        self, experiment_id: str, seed: int
    ) -> dict[str, float] | None:
        """Extract the last logged metrics from metrics.jsonl."""
        metrics_path = self.run_dir(experiment_id, seed) / "metrics.jsonl"
        if not metrics_path.exists():
            return None

        last_line = ""
        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped

        if not last_line:
            return None

        try:
            return json.loads(last_line)
        except json.JSONDecodeError:
            return None
