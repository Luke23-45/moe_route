from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from moe_route.analysis.routing import summarize_routing_run


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run = cfg.get("run")
    if not run:
        raise ValueError("run=... is required.")
    print(json.dumps(summarize_routing_run(run), indent=2))


if __name__ == "__main__":
    main()

