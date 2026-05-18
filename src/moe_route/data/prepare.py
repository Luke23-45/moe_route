from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from moe_route.data.corpus import build_corpus
from moe_route.data.manifest import (
    data_paths,
    dataset_fingerprint,
    prepared_dataset_paths,
    read_manifest,
    write_manifest,
)
from moe_route.data.packing import pack_tokens
from moe_route.tokenization.tokenizers import TextTokenizer


@dataclass(frozen=True)
class PreparedData:
    samples_path: Path
    manifest_path: Path
    num_samples: int
    num_tokens: int
    reused: bool


def _manifest_payload(cfg, tokenizer: TextTokenizer, samples: torch.Tensor | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "fingerprint": dataset_fingerprint(cfg, tokenizer),
        "name": str(cfg.name),
        "adapter": str(cfg.adapter),
        "dataset_name": None if cfg.get("dataset_name") is None else str(cfg.get("dataset_name")),
        "dataset_config": None if cfg.get("dataset_config") is None else str(cfg.get("dataset_config")),
        "split": None if cfg.get("split") is None else str(cfg.get("split")),
        "sequence_length": int(cfg.sequence_length),
        "max_samples": None if cfg.get("max_samples") is None else int(cfg.get("max_samples")),
        "created_at_unix": int(time.time()),
    }
    if samples is not None:
        payload["num_samples"] = int(samples.shape[0])
        payload["num_tokens"] = int(samples.numel())
    return payload


def _is_ready(
    cfg,
    tokenizer: TextTokenizer,
    samples_path: Path,
    manifest_path: Path,
    *,
    honor_rebuild: bool = True,
) -> bool:
    if honor_rebuild and bool(cfg.get("rebuild_cache", False)):
        return False
    if not samples_path.exists() or not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path)
    return manifest.get("fingerprint") == dataset_fingerprint(cfg, tokenizer)


def prepare_data(
    cfg,
    tokenizer: TextTokenizer,
    show_progress: bool = True,
    build_missing: bool = True,
) -> PreparedData:
    """Validate/download/tokenize/pack a configured corpus into a training-ready tensor shard."""
    paths = data_paths(cfg)
    for path in (paths.root_dir, paths.raw_dir, paths.prepared_dir, paths.cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    samples_path, manifest_path = prepared_dataset_paths(cfg, tokenizer)
    if _is_ready(cfg, tokenizer, samples_path, manifest_path, honor_rebuild=build_missing):
        manifest = read_manifest(manifest_path)
        return PreparedData(
            samples_path=samples_path,
            manifest_path=manifest_path,
            num_samples=int(manifest["num_samples"]),
            num_tokens=int(manifest["num_tokens"]),
            reused=True,
        )
    if not build_missing:
        raise FileNotFoundError(
            f"Prepared data for {cfg.name} is missing or stale at {samples_path}. "
            "Run scripts/prepare_data.py or enable trainer.prepare_data=true."
        )

    corpus = build_corpus(cfg)
    token_buffer: list[int] = []
    max_samples = cfg.get("max_samples")
    total = None if max_samples is None else int(max_samples)
    iterator = tqdm(
        corpus.texts(),
        total=total,
        desc=f"prepare:{cfg.name}",
        unit="docs",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for text in iterator:
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=True))
        if show_progress:
            iterator.set_postfix(tokens=len(token_buffer), refresh=False)

    if not token_buffer:
        raise ValueError(f"Data source {cfg.name} produced no tokenizable text.")
    tokens = torch.tensor(token_buffer, dtype=torch.long)
    samples = pack_tokens(tokens, int(cfg.sequence_length))
    torch.save(samples, samples_path)
    payload = _manifest_payload(cfg, tokenizer, samples)
    write_manifest(manifest_path, payload)
    return PreparedData(
        samples_path=samples_path,
        manifest_path=manifest_path,
        num_samples=int(samples.shape[0]),
        num_tokens=int(samples.numel()),
        reused=False,
    )
