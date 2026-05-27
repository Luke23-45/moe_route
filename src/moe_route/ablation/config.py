"""Ablation parameter space definitions.

Defines the exact parameter axes and their value sets for the reflected
mechanism ablation study, as specified in the publication plan:

  - ``pressure_lr ∈ {0.01, 0.05, 0.1}``
  - ``capacity_factor ∈ {0.5, 1.0}``
  - ``shared_experts ∈ {0, 1}``
  - ``pressure_warmup_steps ∈ {0, 1000}``
  - ``learnable_bias ∈ {false, true}``
  - Optional: ``gate_function ∈ {softmax, sigmoid}``

Each axis maps to a specific Hydra override path in the router config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AblationAxis:
    """A single dimension of the ablation parameter space.

    Attributes
    ----------
    name : str
        Human-readable name for this axis (e.g. ``"pressure_lr"``).
    hydra_key : str
        The Hydra override key (e.g. ``"router.pressure_lr"``).
    values : tuple[str, ...]
        The set of values to sweep over, as strings (Hydra override values).
    short_name : str
        Abbreviated name for filesystem identifiers (e.g. ``"plr"``).
    """

    name: str
    hydra_key: str
    values: tuple[str, ...]
    short_name: str


# ---------------------------------------------------------------------------
# Canonical ablation axes from the publication plan
# ---------------------------------------------------------------------------

PRESSURE_LR = AblationAxis(
    name="pressure_lr",
    hydra_key="router.pressure_lr",
    values=("0.01", "0.05", "0.1"),
    short_name="plr",
)

CAPACITY_FACTOR = AblationAxis(
    name="capacity_factor",
    hydra_key="router.capacity_factor",
    values=("0.5", "1.0"),
    short_name="cf",
)

SHARED_EXPERTS = AblationAxis(
    name="shared_experts",
    hydra_key="router.shared_experts",
    values=("0", "1"),
    short_name="shared",
)

PRESSURE_WARMUP = AblationAxis(
    name="pressure_warmup_steps",
    hydra_key="router.pressure_warmup_steps",
    values=("0", "1000"),
    short_name="warmup",
)

LEARNABLE_BIAS = AblationAxis(
    name="learnable_bias",
    hydra_key="router.learnable_bias",
    values=("false", "true"),
    short_name="bias",
)

GATE_FUNCTION = AblationAxis(
    name="gate_function",
    hydra_key="router.gate_function",
    values=("softmax", "sigmoid"),
    short_name="gate",
)

# The mandatory axes per the publication plan
MANDATORY_AXES: tuple[AblationAxis, ...] = (
    PRESSURE_LR,
    CAPACITY_FACTOR,
    SHARED_EXPERTS,
    PRESSURE_WARMUP,
    LEARNABLE_BIAS,
)

# Optional axis — only if stable in code
OPTIONAL_AXES: tuple[AblationAxis, ...] = (
    GATE_FUNCTION,
)


def get_axes(include_optional: bool = False) -> Sequence[AblationAxis]:
    """Return the ablation axes to sweep over.

    Parameters
    ----------
    include_optional : bool
        If True, include the optional ``gate_function`` axis.
    """
    if include_optional:
        return MANDATORY_AXES + OPTIONAL_AXES
    return MANDATORY_AXES
