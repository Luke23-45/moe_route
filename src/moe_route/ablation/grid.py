"""Ablation grid sweep generation.

Generates the full set of ablation parameter combinations and returns them
as lightweight ``AblationPoint`` descriptors.  These descriptors are
decoupled from the experiment orchestration layer (``studies/``) so that
the ``moe_route`` package has no dependency on the studies framework.

The Phase B runner (``studies/runners/run_phase_b.py``) is responsible
for converting ``AblationPoint`` objects into ``ExperimentSpec`` objects.

Sweep strategies:
  - ``"oat"``  — one-at-a-time (default, plan-aligned).  Varies each axis
    independently while holding the others at their defaults.
  - ``"full"`` — full Cartesian product (expensive: 48 experiments for the
    default 5 mandatory axes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from moe_route.ablation.config import AblationAxis, get_axes

# ---------------------------------------------------------------------------
# Default (baseline) values for each reflected knob
# ---------------------------------------------------------------------------
# These match the reflected_sparse.yaml config that the plan designates
# as the paper method.

REFLECTED_DEFAULTS: dict[str, str] = {
    "router.pressure_lr": "0.05",
    "router.capacity_factor": "0.5",
    "router.shared_experts": "0",
    "router.pressure_warmup_steps": "1000",
    "router.learnable_bias": "true",
    "router.gate_function": "sigmoid",
}

# Short-name mapping for building human-readable identifiers
_SHORT_NAMES: dict[str, str] = {
    "pressure_lr": "plr",
    "capacity_factor": "cf",
    "shared_experts": "shared",
    "pressure_warmup_steps": "warmup",
    "learnable_bias": "bias",
    "gate_function": "gate",
}


@dataclass(frozen=True)
class AblationPoint:
    """A single ablation grid point — a lightweight descriptor.

    Attributes
    ----------
    hydra_overrides : tuple[str, ...]
        Hydra override strings (e.g. ``"router.pressure_lr=0.05"``).
    axis_values : dict[str, str]
        Mapping from Hydra key to value for every swept axis.
    experiment_id : str
        Deterministic, filesystem-safe identifier for this point
        (e.g. ``"reflected_plr0.05_cf0.5_shared0_warmup1000_biastrue"``).
    description : str
        Human-readable description.
    tags : dict[str, str]
        Key-value metadata for filtering and reporting.
    is_default : bool
        True if this point represents the default configuration.
    """

    hydra_overrides: tuple[str, ...]
    axis_values: dict[str, str] = field(default_factory=dict)
    experiment_id: str = ""
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    is_default: bool = False


def _build_experiment_id(axis_values: dict[str, str]) -> str:
    """Build a human-readable experiment_id from axis name-value pairs.

    Example: ``"reflected_plr0.05_cf0.5_shared0_warmup1000_biastrue"``
    """
    parts = ["reflected"]
    for key in sorted(axis_values.keys()):
        val = axis_values[key]
        param_name = key.rsplit(".", 1)[-1]
        short = _SHORT_NAMES.get(param_name, param_name)
        parts.append(f"{short}{val}")
    return "_".join(parts)


def _make_point(
    axis_values: dict[str, str],
    description: str,
    is_default: bool = False,
) -> AblationPoint:
    """Create an AblationPoint for a single grid coordinate."""
    overrides = tuple(f"{k}={v}" for k, v in sorted(axis_values.items()))
    exp_id = _build_experiment_id(axis_values)
    tags = {
        "method": "reflected_v2",
        "ablation": "true",
        **{k.rsplit(".", 1)[-1]: v for k, v in axis_values.items()},
    }
    if is_default:
        tags["role"] = "ablation_default"

    return AblationPoint(
        hydra_overrides=overrides,
        axis_values=dict(axis_values),
        experiment_id=exp_id,
        description=description,
        tags=tags,
        is_default=is_default,
    )


def build_oat_grid(
    axes: Sequence[AblationAxis],
    defaults: dict[str, str] | None = None,
) -> list[AblationPoint]:
    """Generate one-at-a-time ablation grid.

    For each axis, varies that axis across all its values while holding
    every other axis at its default.  The default configuration is always
    included exactly once.

    Parameters
    ----------
    axes : Sequence[AblationAxis]
        The ablation axes to sweep.
    defaults : dict[str, str] | None
        Default values for each axis.  If ``None``, uses ``REFLECTED_DEFAULTS``.

    Returns
    -------
    list[AblationPoint]
        Deduplicated list of ablation points.
    """
    defaults = defaults or REFLECTED_DEFAULTS
    points: list[AblationPoint] = []
    seen_ids: set[str] = set()

    # Always include the default configuration as the reference point
    default_values = {a.hydra_key: defaults[a.hydra_key] for a in axes}
    default_point = _make_point(
        default_values, "Reflected v2 default (ablation baseline)", is_default=True
    )
    points.append(default_point)
    seen_ids.add(default_point.experiment_id)

    # For each axis, sweep its values while holding others at default
    for axis in axes:
        for val in axis.values:
            values = dict(default_values)
            values[axis.hydra_key] = val
            desc = f"Ablation: {axis.name}={val} (others at default)"
            point = _make_point(values, desc)
            if point.experiment_id not in seen_ids:
                points.append(point)
                seen_ids.add(point.experiment_id)

    return points


def build_full_grid(
    axes: Sequence[AblationAxis],
) -> list[AblationPoint]:
    """Generate full Cartesian product ablation grid.

    WARNING: This produces ``∏ |axis.values|`` experiments.  For the
    default 5 mandatory axes, that's 3×2×2×2×2 = 48 experiments.
    Use only if compute budget allows.

    Parameters
    ----------
    axes : Sequence[AblationAxis]
        The ablation axes to sweep.

    Returns
    -------
    list[AblationPoint]
        Full factorial ablation points.
    """
    value_lists = [[(a.hydra_key, v) for v in a.values] for a in axes]
    points: list[AblationPoint] = []

    for combo in product(*value_lists):
        values = dict(combo)
        parts = [f"{k.rsplit('.', 1)[-1]}={v}" for k, v in sorted(values.items())]
        desc = f"Ablation (full grid): {', '.join(parts)}"
        points.append(_make_point(values, desc))

    return points


def build_ablation_grid(
    *,
    strategy: str = "oat",
    include_gate_sweep: bool = False,
) -> list[AblationPoint]:
    """Build ablation grid using the specified strategy.

    Parameters
    ----------
    strategy : str
        ``"oat"`` for one-at-a-time (default, plan-aligned) or
        ``"full"`` for full Cartesian product.
    include_gate_sweep : bool
        If True, include the optional ``gate_function`` axis.

    Returns
    -------
    list[AblationPoint]
        The ablation grid points.
    """
    axes = get_axes(include_optional=include_gate_sweep)

    if strategy == "oat":
        return build_oat_grid(axes)
    elif strategy == "full":
        return build_full_grid(axes)
    else:
        raise ValueError(f"Unknown ablation strategy: {strategy!r}. Use 'oat' or 'full'.")
