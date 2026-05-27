"""Pre-flight configuration and compatibility validation.

Runs a comprehensive set of checks *before* any GPU time is spent to catch
config mismatches, missing files, incompatible settings, and override syntax
errors.  Validation failures produce a clear, categorized error report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from studies.utils.experiment_spec import DispatchMode, ExperimentSpec
from studies.utils.seed_manager import ScheduledRun


@dataclass
class ValidationReport:
    """Accumulates validation errors and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def print_report(self) -> None:
        """Print a formatted validation report."""
        if self.passed and not self.warnings:
            print("  [OK] All validation checks passed.\n")
            return

        print(f"\n{'=' * 70}")
        if not self.passed:
            print("  VALIDATION FAILED")
        else:
            print("  VALIDATION PASSED WITH WARNINGS")
        print(f"{'=' * 70}")

        for err in self.errors:
            print(f"  ✗ ERROR:   {err}")
        for warn in self.warnings:
            print(f"  ⚠ WARNING: {warn}")
        print(f"{'=' * 70}\n")

    def raise_if_failed(self) -> None:
        """Raise SystemExit if validation failed."""
        if not self.passed:
            self.print_report()
            raise SystemExit(1)


def validate_override_syntax(overrides: Sequence[str]) -> list[str]:
    """Validate Hydra override syntax. Returns list of error messages."""
    errors: list[str] = []
    for item in overrides:
        if "=" not in item and not item.startswith("+") and not item.startswith("~"):
            errors.append(f"Invalid Hydra override syntax: {item!r}. Use KEY=VALUE format.")
    return errors


def validate_spec(spec: ExperimentSpec, config_root: Path) -> list[str]:
    """Validate a single ExperimentSpec against the config directory.

    Checks:
      1. All referenced config files exist.
      2. MoE experiments have a router config.
      3. Dense experiments don't specify a router.
      4. Dispatch mode is compatible with MoE setting.
      5. Baked-in Hydra overrides have valid syntax.
    """
    errors: list[str] = []
    tag = f"[{spec.experiment_id}]"

    # --- Config file existence ---
    exp_path = config_root / "experiment" / f"{spec.experiment}.yaml"
    if not exp_path.exists():
        errors.append(f"{tag} Experiment config not found: {exp_path}")

    model_path = config_root / "model" / f"{spec.model}.yaml"
    if not model_path.exists():
        errors.append(f"{tag} Model config not found: {model_path}")

    if spec.router is not None:
        router_path = config_root / "router" / f"{spec.router}.yaml"
        if not router_path.exists():
            errors.append(f"{tag} Router config not found: {router_path}")

    data_path = config_root / "data" / f"{spec.data}.yaml"
    if not data_path.exists():
        errors.append(f"{tag} Data config not found: {data_path}")

    trainer_path = config_root / "trainer" / f"{spec.trainer}.yaml"
    if not trainer_path.exists():
        errors.append(f"{tag} Trainer config not found: {trainer_path}")

    # --- MoE / Router compatibility ---
    if spec.is_moe and spec.router is None:
        errors.append(f"{tag} MoE experiment requires a router config, but router is None")

    if not spec.is_moe and spec.router is not None:
        errors.append(f"{tag} Dense experiment should not specify a router (got {spec.router!r})")

    # --- Dispatch mode compatibility ---
    if not spec.is_moe and spec.dispatch_mode != DispatchMode.NONE:
        errors.append(
            f"{tag} Dense experiment dispatch_mode must be NONE, "
            f"got {spec.dispatch_mode.value!r}"
        )

    if spec.is_moe and spec.dispatch_mode == DispatchMode.NONE:
        errors.append(
            f"{tag} MoE experiment dispatch_mode must be SPARSE or DENSE, got NONE"
        )

    # --- Override syntax ---
    override_errors = validate_override_syntax(spec.hydra_overrides)
    for err in override_errors:
        errors.append(f"{tag} {err}")

    return errors


def validate_study(
    runs: Sequence[ScheduledRun],
    config_root: Path,
    study_name: str,
) -> ValidationReport:
    """Validate an entire study's worth of scheduled runs.

    Parameters
    ----------
    runs : Sequence[ScheduledRun]
        All runs scheduled for this study.
    config_root : Path
        Path to the configs/ directory.
    study_name : str
        Name of the study for reporting.

    Returns
    -------
    ValidationReport
        Report with all errors and warnings.
    """
    report = ValidationReport()

    if not runs:
        report.add_error(f"Study {study_name!r} has no scheduled runs.")
        return report

    # Validate each unique spec (avoid re-validating the same spec for each seed)
    validated_specs: set[str] = set()
    for run in runs:
        spec_id = run.spec.experiment_id
        if spec_id not in validated_specs:
            spec_errors = validate_spec(run.spec, config_root)
            for err in spec_errors:
                report.add_error(err)
            validated_specs.add(spec_id)

    # Check for duplicate run names
    seen_names: dict[str, int] = {}
    for run in runs:
        seen_names[run.run_name] = seen_names.get(run.run_name, 0) + 1
    for name, count in seen_names.items():
        if count > 1:
            report.add_error(f"Duplicate run_name {name!r} appears {count} times.")

    # Check config_root exists
    if not config_root.exists():
        report.add_error(f"Config root directory does not exist: {config_root}")

    # Warn if running many experiments
    if len(runs) > 50:
        report.add_warning(
            f"Study {study_name!r} has {len(runs)} runs scheduled. "
            f"This may take a very long time."
        )

    return report
