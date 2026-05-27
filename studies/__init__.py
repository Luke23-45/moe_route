"""Experiment studies orchestration package for Reflected MoE v2.

This package provides a structured framework for running the three-phase
experiment program defined in the publication plan:

  Phase A: TinyStories baseline comparison (multi-seed)
  Phase B: Reflected mechanism ablations (TinyStories only)
  Phase C: FineWeb scale-up baseline comparison (multi-seed)

Usage:
  python -m studies.runners.run_phase_a [--dry-run] [--smoke] [--only METHOD]
  python -m studies.runners.run_phase_b [--dry-run] [--smoke]
  python -m studies.runners.run_phase_c [--dry-run] [--smoke]
"""
