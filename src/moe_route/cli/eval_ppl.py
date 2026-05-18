from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from moe_route.evaluation.ppl import evaluate_perplexity, load_cfg_from_checkpoint


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    eval_cfg = load_cfg_from_checkpoint(checkpoint) if checkpoint else cfg
    metrics = evaluate_perplexity(eval_cfg, checkpoint)
    output_file = cfg.get("output_file")
    if output_file:
        from pathlib import Path
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved PPL metrics to {output_file}")
    else:
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

