from __future__ import annotations

import hydra
from omegaconf import DictConfig

from moe_route.training.trainer import train


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

