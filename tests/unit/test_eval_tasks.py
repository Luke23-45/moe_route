from __future__ import annotations

import pytest
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


def test_evaluate_tasks_rejects_native_checkpoint_for_hf_adapter() -> None:
    cfg = OmegaConf.create(
        {
            "eval": {
                "tasks": ["hellaswag"],
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
    with pytest.raises(ValueError, match="does not support native training checkpoints directly"):
        evaluate_tasks(cfg, "model.pt")
