from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from moe_route.data.prepare import prepare_data
from moe_route.tokenization.tokenizers import build_tokenizer


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    tokenizer = build_tokenizer(cfg.tokenizer)
    prepared = prepare_data(cfg.data, tokenizer, show_progress=True)
    print(
        json.dumps(
            {
                "samples_path": str(prepared.samples_path),
                "manifest_path": str(prepared.manifest_path),
                "num_samples": prepared.num_samples,
                "num_tokens": prepared.num_tokens,
                "reused": prepared.reused,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

