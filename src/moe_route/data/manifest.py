from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from moe_route.tokenization.tokenizers import TextTokenizer


@dataclass(frozen=True)
class DataPaths:
    root_dir: Path
    raw_dir: Path
    prepared_dir: Path
    cache_dir: Path


def data_paths(cfg) -> DataPaths:
    root = Path(str(cfg.get("root_dir", Path("artifacts") / "data" / str(cfg.name))))
    raw = Path(str(cfg.get("raw_dir", root / "raw")))
    prepared = Path(str(cfg.get("prepared_dir", root / "prepared")))
    cache = Path(str(cfg.get("cache_dir", prepared / "cache")))
    return DataPaths(root, raw, prepared, cache)


def tokenizer_fingerprint(tokenizer: TextTokenizer) -> str:
    fields = {
        "class": tokenizer.__class__.__name__,
        "vocab_size": tokenizer.vocab_size,
        "pad": tokenizer.pad_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(cfg, tokenizer: TextTokenizer) -> str:
    fields: dict[str, Any] = {
        "name": cfg.name,
        "adapter": cfg.adapter,
        "dataset_name": cfg.get("dataset_name"),
        "dataset_config": cfg.get("dataset_config"),
        "split": cfg.get("split"),
        "streaming": bool(cfg.get("streaming", False)),
        "max_samples": cfg.get("max_samples"),
        "sequence_length": int(cfg.sequence_length),
        "text_field": cfg.get("text_field"),
        "tokenizer": tokenizer_fingerprint(tokenizer),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def prepared_dataset_paths(cfg, tokenizer: TextTokenizer) -> tuple[Path, Path]:
    paths = data_paths(cfg)
    key = dataset_fingerprint(cfg, tokenizer)[:24]
    return paths.prepared_dir / f"packed_{key}.bin", paths.prepared_dir / f"packed_{key}.json"


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
