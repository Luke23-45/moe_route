"""Central experiment specification dataclass and registry.

Provides a self-documenting, immutable record for every experiment variant
used across the three study phases. Extends the pattern established in
``scripts/launch_tinystories_suite.py`` with explicit capacity-factor
tagging and phase membership so that experiment specifications can be
filtered, validated, and composed programmatically by the phase runners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique


@unique
class DispatchMode(Enum):
    """Token dispatch strategy used by the MoE layer."""

    SPARSE = "sparse"
    DENSE = "dense"
    NONE = "none"  # standard FFN, no MoE


@unique
class Phase(Enum):
    """Experiment phase as defined in the publication plan."""

    A = "phase_a"
    B = "phase_b"
    C = "phase_c"


@dataclass(frozen=True)
class ExperimentSpec:
    """Self-documenting experiment specification with full validation metadata.

    Each spec uniquely identifies a *method variant* within a study phase.
    A single spec can be run across multiple seeds (see ``seed_manager``).

    Attributes
    ----------
    name : str
        Short CLI key used with ``--only`` (e.g. ``"reflected_sparse"``).
    experiment : str
        Hydra experiment config name (without ``.yaml``).
    model : str
        Hydra model config name (without ``.yaml``).
    router : str | None
        Hydra router config name. ``None`` for dense baselines.
    data : str
        Hydra data config name (without ``.yaml``).
    trainer : str
        Hydra trainer config name (without ``.yaml``).
    is_moe : bool
        Whether this experiment uses MoE layers.
    dispatch_mode : DispatchMode
        Token dispatch strategy.
    description : str
        Human-readable description shown in plan printouts.
    phases : frozenset[Phase]
        Which study phases this spec participates in.
    is_control : bool
        ``True`` for supporting controls (Dense, Top-1) that receive
        fewer seeds than the main methods.
    is_reference_baseline : bool
        ``True`` for baselines (Top-2, DeepSeek-LFB) included in the
        ablation study (Phase B) as reference points only.
    hydra_overrides : tuple[str, ...]
        Extra Hydra overrides baked into this spec. Used to set router
        knobs like ``router.capacity_factor=0.5`` without duplicating
        the full router YAML for every capacity variant.
    tags : dict[str, str]
        Arbitrary key-value metadata for filtering and reporting.
    """

    name: str
    experiment: str
    model: str
    router: str | None = None
    data: str = "tinystories"
    trainer: str = "tinystories"
    is_moe: bool = False
    dispatch_mode: DispatchMode = DispatchMode.NONE
    description: str = ""
    phases: frozenset[Phase] = frozenset()
    is_control: bool = False
    is_reference_baseline: bool = False
    hydra_overrides: tuple[str, ...] = ()
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def experiment_id(self) -> str:
        """Deterministic, filesystem-safe identifier for this variant.

        Encodes the method name and any baked-in parameter overrides so that
        two specs that differ only in ``router.capacity_factor`` produce
        distinct directories in the persistence hierarchy.
        """
        if self.tags.get("ablation") == "true":
            return self.name

        parts = [self.name]
        for override in sorted(self.hydra_overrides):
            # Extract the value portion: "router.capacity_factor=0.5" → "cf0.5"
            if "capacity_factor" in override:
                val = override.split("=", 1)[1]
                parts.append(f"cf{val}")
            elif "=" in override:
                key, val = override.split("=", 1)
                short_key = key.rsplit(".", 1)[-1]
                parts.append(f"{short_key}{val}")
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Phase A — TinyStories Baseline Comparison
# ---------------------------------------------------------------------------

# Main methods run at two capacity factors: 0.5 and 1.0.
# Supporting controls (Dense, Top-1) run once at the default capacity.

_PHASE_A_MAIN_METHODS: list[ExperimentSpec] = []

for _cf in ("0.5", "1.0"):
    _cf_override = f"router.capacity_factor={_cf}"

    _PHASE_A_MAIN_METHODS.extend([
        ExperimentSpec(
            name="top2",
            experiment="tinystories_top2",
            model="tiny_moe",
            router="top2",
            is_moe=True,
            dispatch_mode=DispatchMode.SPARSE,
            description=f"Top-2 sparse MoE routing (cf={_cf})",
            phases=frozenset({Phase.A}),
            hydra_overrides=(_cf_override,),
            tags={"method": "top2", "capacity_factor": _cf},
        ),
        ExperimentSpec(
            name="deepseek_lfb",
            experiment="tinystories_deepseek_lfb",
            model="tiny_moe",
            router="deepseek_lfb",
            is_moe=True,
            dispatch_mode=DispatchMode.SPARSE,
            description=f"DeepSeek Loss-Free Balancing (cf={_cf})",
            phases=frozenset({Phase.A}),
            hydra_overrides=(_cf_override,),
            tags={"method": "deepseek_lfb", "capacity_factor": _cf},
        ),
        ExperimentSpec(
            name="reflected_sparse",
            experiment="tinystories_reflected_v2",
            model="tiny_moe",
            router="reflected_sparse",
            is_moe=True,
            dispatch_mode=DispatchMode.SPARSE,
            description=f"Reflected v2 sparse top-2 (cf={_cf})",
            phases=frozenset({Phase.A}),
            hydra_overrides=(_cf_override,),
            tags={"method": "reflected_v2", "capacity_factor": _cf},
        ),
    ])

_PHASE_A_CONTROLS: list[ExperimentSpec] = [
    ExperimentSpec(
        name="dense",
        experiment="tinystories_dense",
        model="tiny_dense",
        router=None,
        is_moe=False,
        dispatch_mode=DispatchMode.NONE,
        description="Dense baseline (standard FFN, no MoE)",
        phases=frozenset({Phase.A}),
        is_control=True,
        tags={"method": "dense"},
    ),
    ExperimentSpec(
        name="top1",
        experiment="tinystories_top1",
        model="tiny_moe",
        router="top1",
        is_moe=True,
        dispatch_mode=DispatchMode.SPARSE,
        description="Top-1 sparse MoE routing (supporting control)",
        phases=frozenset({Phase.A}),
        is_control=True,
        tags={"method": "top1"},
    ),
]

PHASE_A_SPECS: tuple[ExperimentSpec, ...] = tuple(_PHASE_A_MAIN_METHODS + _PHASE_A_CONTROLS)

# ---------------------------------------------------------------------------
# Phase B — Reflected Ablation Reference Baselines
# ---------------------------------------------------------------------------
# The ablation study only varies reflected knobs.  Top-2 and DeepSeek-LFB
# are included as *reference baselines* so each reflected variant can be
# compared against strong non-reflected alternatives.

PHASE_B_REFERENCE_BASELINES: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="top2",
        experiment="tinystories_top2",
        model="tiny_moe",
        router="top2",
        is_moe=True,
        dispatch_mode=DispatchMode.SPARSE,
        description="Top-2 reference baseline for ablation study",
        phases=frozenset({Phase.B}),
        is_reference_baseline=True,
        tags={"method": "top2", "role": "reference_baseline"},
    ),
    ExperimentSpec(
        name="deepseek_lfb",
        experiment="tinystories_deepseek_lfb",
        model="tiny_moe",
        router="deepseek_lfb",
        is_moe=True,
        dispatch_mode=DispatchMode.SPARSE,
        description="DeepSeek-LFB reference baseline for ablation study",
        phases=frozenset({Phase.B}),
        is_reference_baseline=True,
        tags={"method": "deepseek_lfb", "role": "reference_baseline"},
    ),
)

# ---------------------------------------------------------------------------
# Phase C — FineWeb Baseline Comparison
# ---------------------------------------------------------------------------

_PHASE_C_SPECS: list[ExperimentSpec] = []

for _cf in ("0.5", "1.0"):
    _cf_override = f"router.capacity_factor={_cf}"

    _PHASE_C_SPECS.extend([
        ExperimentSpec(
            name="reflected_sparse",
            experiment="fineweb_reflected_v2",
            model="tiny_moe",
            router="reflected_sparse",
            data="fineweb_10bt",
            trainer="fineweb",
            is_moe=True,
            dispatch_mode=DispatchMode.SPARSE,
            description=f"Reflected v2 on FineWeb (cf={_cf})",
            phases=frozenset({Phase.C}),
            hydra_overrides=(_cf_override,),
            tags={"method": "reflected_v2", "capacity_factor": _cf},
        ),
        ExperimentSpec(
            name="deepseek_lfb",
            experiment="fineweb_deepseek_lfb",
            model="tiny_moe",
            router="deepseek_lfb",
            data="fineweb_10bt",
            trainer="fineweb",
            is_moe=True,
            dispatch_mode=DispatchMode.SPARSE,
            description=f"DeepSeek-LFB on FineWeb (cf={_cf})",
            phases=frozenset({Phase.C}),
            hydra_overrides=(_cf_override,),
            tags={"method": "deepseek_lfb", "capacity_factor": _cf},
        ),
        ExperimentSpec(
            name="top2",
            experiment="fineweb_top2",
            model="tiny_moe",
            router="top2",
            data="fineweb_10bt",
            trainer="fineweb",
            is_moe=True,
            dispatch_mode=DispatchMode.SPARSE,
            description=f"Top-2 anchor on FineWeb (cf={_cf})",
            phases=frozenset({Phase.C}),
            is_control=True,
            hydra_overrides=(_cf_override,),
            tags={"method": "top2", "capacity_factor": _cf, "role": "anchor"},
        ),
    ])

PHASE_C_SPECS: tuple[ExperimentSpec, ...] = tuple(_PHASE_C_SPECS)
