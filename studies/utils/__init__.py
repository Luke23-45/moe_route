"""Shared utilities for experiment orchestration.

Modules:
  experiment_spec   – Central ExperimentSpec dataclass and registry
  seed_manager      – Multi-seed scheduling with deterministic derivation
  persistence       – Hierarchical result storage and aggregation
  manifest          – Per-run experiment manifest capture
  validation        – Pre-flight config and compatibility validation
  runner_base       – Base orchestrator class for phase runners
"""
