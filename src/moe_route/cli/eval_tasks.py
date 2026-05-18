from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from moe_route.evaluation.tasks import evaluate_tasks


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("checkpoint=... is required for downstream task evaluation.")
    print(json.dumps(evaluate_tasks(cfg, checkpoint), indent=2))


if __name__ == "__main__":
    main()

