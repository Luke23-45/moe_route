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
    results = evaluate_tasks(cfg, checkpoint)
    output_file = cfg.get("output_file")
    if output_file:
        from pathlib import Path
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved Tasks metrics to {output_file}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

