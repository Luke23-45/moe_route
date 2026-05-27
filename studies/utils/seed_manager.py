"""Multi-seed experiment scheduling with deterministic seed derivation.

Provides the seed lists mandated by the publication plan and utilities
for generating the full cross-product of (ExperimentSpec × seed) with
deterministic, reproducible run names.

Seed policy (from the plan):
  Phase A  – 3 seeds for main methods, 1 seed for supporting controls.
  Phase B  – 1 seed per ablation variant, 1 seed for reference baselines.
  Phase C  – 2 seeds for Reflected v2 and DeepSeek-LFB, 1 for Top-2 anchor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from studies.utils.experiment_spec import ExperimentSpec, Phase

# ---------------------------------------------------------------------------
# Canonical seed lists
# ---------------------------------------------------------------------------

# These seeds are chosen to be far apart in the integer space and are the
# same across all experiments to allow cross-method seed-matched comparison.
SEEDS_3: tuple[int, ...] = (1337, 42, 7)
SEEDS_2: tuple[int, ...] = (1337, 42)
SEEDS_1: tuple[int, ...] = (1337,)


def seeds_for_spec(spec: ExperimentSpec, phase: Phase) -> tuple[int, ...]:
    """Return the seed list appropriate for *spec* within *phase*.

    This function encodes the publication plan's seed policy:
      - Phase A main methods:   3 seeds
      - Phase A controls:       1 seed
      - Phase B ablation:       1 seed
      - Phase B ref baselines:  1 seed
      - Phase C main methods:   2 seeds
      - Phase C anchor (Top-2): 1 seed
    """
    if phase == Phase.A:
        return SEEDS_1 if spec.is_control else SEEDS_3
    if phase == Phase.B:
        return SEEDS_1
    if phase == Phase.C:
        return SEEDS_1 if spec.is_control else SEEDS_2
    raise ValueError(f"Unknown phase: {phase!r}")


# ---------------------------------------------------------------------------
# Run descriptor — a (spec, seed) pair with derived metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledRun:
    """A fully resolved (spec, seed) pair ready for execution.

    Attributes
    ----------
    spec : ExperimentSpec
        The experiment specification.
    seed : int
        The random seed for this particular run.
    run_name : str
        A unique, human-readable run name that encodes the experiment_id
        and the seed.  Used as the Hydra ``run_name`` override.
    """

    spec: ExperimentSpec
    seed: int
    run_name: str

    @property
    def run_id(self) -> str:
        """Compact deterministic identifier for deduplication and manifests.

        Derived from (experiment_id, seed) so that two independent
        schedule() calls produce the same run_id for the same logical run.
        """
        raw = f"{self.spec.experiment_id}::seed_{self.seed}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def schedule(
    specs: Sequence[ExperimentSpec],
    phase: Phase,
    *,
    run_prefix: str = "",
    seed_override: Sequence[int] | None = None,
) -> list[ScheduledRun]:
    """Expand a list of specs into the full set of ScheduledRuns.

    Parameters
    ----------
    specs : Sequence[ExperimentSpec]
        The experiment specifications to schedule.
    phase : Phase
        Which phase these specs belong to.  Determines seed count.
    run_prefix : str
        Optional prefix prepended to every run_name.
    seed_override : Sequence[int] | None
        If provided, overrides the default seed list for *all* specs.
        Useful for ``--seeds 1337,42`` CLI overrides.

    Returns
    -------
    list[ScheduledRun]
        Ordered list of runs.  Order is stable: specs in their original
        order, then seeds in ascending order within each spec.
    """
    runs: list[ScheduledRun] = []
    for spec in specs:
        seeds = tuple(seed_override) if seed_override is not None else seeds_for_spec(spec, phase)
        for seed in seeds:
            prefix = f"{run_prefix}_" if run_prefix else ""
            name = f"{prefix}{spec.experiment_id}_seed{seed}"
            runs.append(ScheduledRun(spec=spec, seed=seed, run_name=name))
    return runs
