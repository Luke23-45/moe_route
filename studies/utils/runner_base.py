"""Base orchestrator class for phase runners.

Provides ``PhaseRunner`` — the single class that all phase runner scripts
use to orchestrate multi-seed experiment execution.  Handles:

  - Pre-flight validation
  - Data preparation (once per phase)
  - Sequential execution of (spec, seed) runs via subprocess
  - Post-run result collection into the persistence layer
  - Cross-seed aggregation and study summary generation
  - Execution plan printing and dry-run mode
  - Graceful failure handling with partial result preservation
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from studies.utils.experiment_spec import ExperimentSpec
from studies.utils.manifest import RunManifest
from studies.utils.persistence import ResultStore
from studies.utils.seed_manager import Phase, ScheduledRun, schedule
from studies.utils.validation import validate_study

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs"
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_RESULT_ROOT = REPO_ROOT / "artifacts" / "results"


def _python_env() -> dict[str, str]:
    """Build an environment dict with PYTHONPATH pointing to src/."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    return env


def _module_cmd(module: str, overrides: list[str]) -> list[str]:
    """Build a ``python -m module`` command with Hydra overrides."""
    return [sys.executable, "-m", module, *overrides]


class PhaseRunner:
    """Orchestrates multi-seed experiment execution for a study phase.

    Parameters
    ----------
    study_name : str
        Human-readable study name (e.g. ``"phase_a_tinystories_baseline"``).
        Used as the top-level directory in the persistence hierarchy.
    phase : Phase
        Which phase this runner covers.  Determines seed counts.
    specs : Sequence[ExperimentSpec]
        The experiment specifications to run.
    result_root : Path | str
        Root directory for result persistence.
    run_prefix : str
        Optional prefix for run names.
    """

    def __init__(
        self,
        study_name: str,
        phase: Phase,
        specs: Sequence[ExperimentSpec],
        *,
        result_root: Path | str = DEFAULT_RESULT_ROOT,
        run_prefix: str = "",
    ) -> None:
        self.study_name = study_name
        self.phase = phase
        self.specs = list(specs)
        self.run_prefix = run_prefix
        self.store = ResultStore(result_root, study_name)

    # -----------------------------------------------------------------
    # Execution plan
    # -----------------------------------------------------------------

    def build_schedule(
        self,
        *,
        only: list[str] | None = None,
        seed_override: Sequence[int] | None = None,
    ) -> list[ScheduledRun]:
        """Build the full schedule of runs, optionally filtered.

        Parameters
        ----------
        only : list[str] | None
            If provided, only include specs whose ``name`` is in this list.
        seed_override : Sequence[int] | None
            If provided, overrides the default seed list for all specs.
        """
        selected = self.specs
        if only:
            selected = [s for s in self.specs if s.name in only]
        return schedule(selected, self.phase, run_prefix=self.run_prefix, seed_override=seed_override)

    def print_plan(self, runs: list[ScheduledRun]) -> None:
        """Print a human-readable execution plan."""
        n_specs = len({r.spec.experiment_id for r in runs})
        n_seeds = len({r.seed for r in runs})

        print(f"\n{'=' * 78}")
        print(f"  Study: {self.study_name}")
        print(f"  Phase: {self.phase.value}")
        print(f"  Total runs: {len(runs)}  ({n_specs} experiments × up to {n_seeds} seeds)")
        print(f"{'=' * 78}")

        current_exp_id = None
        for run in runs:
            if run.spec.experiment_id != current_exp_id:
                current_exp_id = run.spec.experiment_id
                mode = run.spec.dispatch_mode.value.upper()
                router = run.spec.router or "(none)"
                print(
                    f"\n  {run.spec.experiment_id:<35}  "
                    f"model={run.spec.model:<12}  "
                    f"router={router:<16}  "
                    f"mode={mode}"
                )
                if run.spec.description:
                    print(f"    {run.spec.description}")
            print(f"    └── seed={run.seed}  run_name={run.run_name}")

        print(f"\n{'=' * 78}\n")

    # -----------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------

    def prepare_data(
        self,
        data_config: str,
        *,
        rebuild: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Prepare data once before the experiment suite starts.

        Parameters
        ----------
        data_config : str
            Hydra data config name (e.g. ``"tinystories"``).
        rebuild : bool
            If True, force rebuild of the data cache.
        dry_run : bool
            If True, only print the command.
        """
        overrides = [f"data={data_config}"]
        if rebuild:
            overrides.append("data.rebuild_cache=true")
        cmd = _module_cmd("moe_route.cli.prepare_data", overrides)
        self._run_subprocess(cmd, "prepare_data", dry_run=dry_run)

    # -----------------------------------------------------------------
    # Core execution
    # -----------------------------------------------------------------

    def build_overrides(
        self,
        run: ScheduledRun,
        *,
        checkpoint_root: str | Path,
        common_overrides: list[str] | None = None,
        skip_prepare: bool = True,
    ) -> list[str]:
        """Build the full Hydra override list for a single run.

        The seed is injected as a top-level override ``seed=<value>`` which
        is picked up by ``seed_everything()`` in the trainer.  This ensures
        every run uses its assigned seed deterministically.
        """
        spec = run.spec
        ckpt_dir = Path(checkpoint_root) / run.run_name

        overrides = [
            f"experiment={spec.experiment}",
            f"model={spec.model}",
            f"data={spec.data}",
            f"trainer={spec.trainer}",
            f"run_name={run.run_name}",
            f"seed={run.seed}",
            f"trainer.save_dir={ckpt_dir}",
            f"trainer.prepare_data={'false' if skip_prepare else 'true'}",
        ]

        if spec.router is not None:
            overrides.append(f"router={spec.router}")

        # Baked-in spec overrides (e.g. capacity_factor variants)
        overrides.extend(spec.hydra_overrides)

        # Common overrides from CLI
        if common_overrides:
            overrides.extend(common_overrides)

        return overrides

    def execute(
        self,
        runs: list[ScheduledRun],
        *,
        checkpoint_root: str | Path = "artifacts/checkpoints",
        common_overrides: list[str] | None = None,
        nproc_per_node: int = 1,
        skip_prepare: bool = True,
        dry_run: bool = False,
        smoke: bool = False,
    ) -> list[tuple[ScheduledRun, bool]]:
        """Execute all scheduled runs sequentially.

        Parameters
        ----------
        runs : list[ScheduledRun]
            The runs to execute (from ``build_schedule()``).
        checkpoint_root : str | Path
            Root directory for checkpoints.
        common_overrides : list[str] | None
            Extra Hydra overrides applied to every run.
        nproc_per_node : int
            Number of processes for distributed training.
        skip_prepare : bool
            If True, assumes data was prepared separately.
        dry_run : bool
            If True, only print commands without executing.
        smoke : bool
            If True, apply smoke-test overrides (CPU, few steps).

        Returns
        -------
        list[tuple[ScheduledRun, bool]]
            List of (run, success) tuples.
        """
        all_overrides = list(common_overrides or [])

        if smoke:
            all_overrides.extend([
                "data=tinystories_smoke",
                "trainer=smoke",
            ])
            # Add max_steps if not already set
            if not any(o.startswith("trainer.max_steps=") for o in all_overrides):
                all_overrides.append("trainer.max_steps=20")

        results: list[tuple[ScheduledRun, bool]] = []

        for i, run in enumerate(runs, 1):
            exp_id = run.spec.experiment_id
            print(
                f"\n[{self.study_name}] ({i}/{len(runs)}) "
                f"Starting: {exp_id} seed={run.seed}"
            )
            print(f"  {run.spec.description}")

            # Create manifest at run start
            manifest = RunManifest.create(
                spec=run.spec,
                seed=run.seed,
                run_name=run.run_name,
                study_name=self.study_name,
            )

            # Save partial manifest (marks run as "running")
            self.store.save_manifest(manifest, exp_id, run.seed)

            overrides = self.build_overrides(
                run,
                checkpoint_root=checkpoint_root,
                common_overrides=all_overrides,
                skip_prepare=skip_prepare,
            )

            # Build command
            if nproc_per_node > 1:
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc-per-node={nproc_per_node}",
                    "-m",
                    "moe_route.cli.train",
                    "distributed=ddp",
                    *overrides,
                ]
            else:
                cmd = _module_cmd("moe_route.cli.train", overrides)

            # Execute
            start_time = time.monotonic()
            exit_code = self._run_subprocess(cmd, run.run_name, dry_run=dry_run)
            elapsed = time.monotonic() - start_time
            success = exit_code == 0

            if not dry_run:
                # Finalize manifest
                manifest.finalize(success=success)
                manifest.wall_clock_seconds = elapsed

                # Locate the training artifact directory
                run_artifact_dir = (
                    Path("artifacts") / "runs" / run.run_name
                )
                checkpoint_path = self._find_latest_checkpoint(
                    Path(checkpoint_root) / run.run_name
                )

                # Collect results into persistence
                self.store.collect_run_results(
                    experiment_id=exp_id,
                    seed=run.seed,
                    manifest=manifest,
                    run_artifact_dir=run_artifact_dir,
                    checkpoint_path=checkpoint_path,
                )

            results.append((run, success))

            if success:
                print(
                    f"[{self.study_name}] ✓ {exp_id} seed={run.seed} "
                    f"completed in {elapsed:.1f}s"
                )
            else:
                print(
                    f"[{self.study_name}] ✗ {exp_id} seed={run.seed} "
                    f"FAILED (exit code {exit_code})"
                )

        return results

    # -----------------------------------------------------------------
    # Post-execution
    # -----------------------------------------------------------------

    def aggregate_and_summarize(self, runs: list[ScheduledRun]) -> dict:
        """Aggregate results across seeds and generate study summary.

        Should be called after ``execute()`` completes.
        """
        # Aggregate each unique experiment
        experiment_ids = sorted({r.spec.experiment_id for r in runs})
        for exp_id in experiment_ids:
            self.store.aggregate_experiment(exp_id)

        # Generate study-level summary
        summary = self.store.generate_study_summary()

        print(f"\n[{self.study_name}] Study summary written to:")
        print(f"  {self.store.study_dir / 'study_summary.json'}")
        return summary

    def print_results_table(self, runs: list[ScheduledRun]) -> None:
        """Print a formatted results table to stdout."""
        experiment_ids = sorted({r.spec.experiment_id for r in runs})

        print(f"\n{'=' * 78}")
        print(f"  Results: {self.study_name}")
        print(f"{'=' * 78}")
        print(f"  {'Experiment':<35}  {'Seeds':>5}  {'Loss (mean±std)':>20}  {'tok/s':>10}")
        print(f"  {'-' * 35}  {'-' * 5}  {'-' * 20}  {'-' * 10}")

        for exp_id in experiment_ids:
            agg_path = self.store.experiment_dir(exp_id) / "aggregate.json"
            if not agg_path.exists():
                print(f"  {exp_id:<35}  {'?':>5}  {'(no data)':>20}")
                continue

            agg = json.loads(agg_path.read_text(encoding="utf-8"))
            n = agg.get("n_seeds", 0)
            means = agg.get("mean", {})
            stds = agg.get("std", {})

            loss_mean = means.get("final_train_loss", float("nan"))
            loss_std = stds.get("final_train_loss", 0.0)
            tps = means.get("tokens_per_second", float("nan"))

            if n > 1:
                loss_str = f"{loss_mean:.4f} ± {loss_std:.4f}"
            else:
                loss_str = f"{loss_mean:.4f}"

            print(f"  {exp_id:<35}  {n:>5}  {loss_str:>20}  {tps:>10.0f}")

        print(f"{'=' * 78}\n")

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _run_subprocess(
        self,
        cmd: list[str],
        label: str,
        *,
        dry_run: bool = False,
    ) -> int:
        """Execute a subprocess with proper environment setup."""
        rendered = " ".join(cmd)
        print(f"[runner] cwd={REPO_ROOT}")
        print(f"[runner] {label}: {rendered}")
        if dry_run:
            return 0
        return subprocess.call(cmd, cwd=str(REPO_ROOT), env=_python_env())

    @staticmethod
    def _find_latest_checkpoint(ckpt_dir: Path) -> str | None:
        """Find the most recent checkpoint file in a directory."""
        if not ckpt_dir.exists():
            return None
        checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None
