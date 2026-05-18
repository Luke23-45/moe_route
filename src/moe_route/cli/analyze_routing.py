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
    results = summarize_routing_run(run)
    output_file = cfg.get("output_file")
    if output_file:
        from pathlib import Path
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved Routing analysis to {output_file}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

