from __future__ import annotations

from omegaconf import OmegaConf

from moe_route.evaluation.tasks import evaluate_tasks


def test_evaluate_tasks_no_tasks_is_noop() -> None:
    cfg = OmegaConf.create(
        {
            "eval": {
                "tasks": [],
                "task_adapter": {
                    "backend": "lm_eval",
                    "model": "hf",
                    "model_args": "",
                    "batch_size": "auto",
                    "limit": None,
                    "output_path": "unused.json",
                },
            }
        }
    )
    assert evaluate_tasks(cfg, "checkpoint") == {}
