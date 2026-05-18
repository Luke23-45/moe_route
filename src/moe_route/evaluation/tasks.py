from __future__ import annotations


def evaluate_tasks(_cfg, _checkpoint: str) -> dict[str, float]:
    raise NotImplementedError(
        "Downstream task evaluation is intentionally adapter-only in v1. "
        "Install the lm-eval extra and wire task names through configs before running real benchmarks."
    )

