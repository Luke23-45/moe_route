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
from moe_route.tokenization.tokenizers import TextTokenizer

TOKEN_DTYPE = torch.int32


@dataclass(frozen=True)
class PreparedData:
    samples_path: Path
    manifest_path: Path
    num_samples: int
    num_tokens: int
    reused: bool
    sequence_length: int


def _manifest_payload(
    cfg,
    tokenizer: TextTokenizer,
    *,
    num_samples: int,
    num_tokens: int,
) -> dict[str, object]:
    return {
        "fingerprint": dataset_fingerprint(cfg, tokenizer),
        "name": str(cfg.name),
        "adapter": str(cfg.adapter),
        "dataset_name": None if cfg.get("dataset_name") is None else str(cfg.get("dataset_name")),
        "dataset_config": None if cfg.get("dataset_config") is None else str(cfg.get("dataset_config")),
        "split": None if cfg.get("split") is None else str(cfg.get("split")),
        "sequence_length": int(cfg.sequence_length),
        "max_samples": None if cfg.get("max_samples") is None else int(cfg.get("max_samples")),
        "num_samples": int(num_samples),
        "num_tokens": int(num_tokens),
        "token_dtype": str(TOKEN_DTYPE).replace("torch.", ""),
        "created_at_unix": int(time.time()),
    }


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


def _flush_buffer(out_file, token_buffer: list[int], width: int) -> tuple[int, int]:
    usable = (len(token_buffer) // width) * width
    if usable == 0:
        return 0, 0
    block = torch.tensor(token_buffer[:usable], dtype=TOKEN_DTYPE)
    block.numpy().tofile(out_file)
    del token_buffer[:usable]
    return usable // width, usable


def prepare_data(
    cfg,
    tokenizer: TextTokenizer,
    show_progress: bool = True,
    build_missing: bool = True,
) -> PreparedData:
    """Prepare packed token sequences without materializing the full corpus in RAM."""
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
            sequence_length=int(manifest["sequence_length"]),
        )
    if not build_missing:
        raise FileNotFoundError(
            f"Prepared data for {cfg.name} is missing or stale at {samples_path}. "
            "Run scripts/prepare_data.py or enable trainer.prepare_data=true."
        )

    corpus = build_corpus(cfg)
    width = int(cfg.sequence_length) + 1
    flush_threshold_sequences = int(cfg.get("prepare_chunk_sequences", 8192))
    flush_threshold_tokens = flush_threshold_sequences * width
    max_samples = cfg.get("max_samples")
    total_docs = None if max_samples is None else int(max_samples)
    token_buffer: list[int] = []
    num_samples = 0
    num_tokens = 0

    iterator = tqdm(
        corpus.texts(),
        total=total_docs,
        desc=f"prepare:{cfg.name}",
        unit="docs",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with samples_path.open("wb") as out_file:
        for text in iterator:
            token_buffer.extend(tokenizer.encode(text, add_special_tokens=True))
            if len(token_buffer) >= flush_threshold_tokens:
                written_samples, written_tokens = _flush_buffer(out_file, token_buffer, width)
                num_samples += written_samples
                num_tokens += written_tokens
            if show_progress:
                iterator.set_postfix(samples=num_samples, buffered=len(token_buffer), refresh=False)

        written_samples, written_tokens = _flush_buffer(out_file, token_buffer, width)
        num_samples += written_samples
        num_tokens += written_tokens

    if num_samples == 0:
        if samples_path.exists():
            samples_path.unlink()
        raise ValueError(f"Data source {cfg.name} produced too few tokens to create one packed sample.")

    payload = _manifest_payload(cfg, tokenizer, num_samples=num_samples, num_tokens=num_tokens)
    write_manifest(manifest_path, payload)
    return PreparedData(
        samples_path=samples_path,
        manifest_path=manifest_path,
        num_samples=num_samples,
        num_tokens=num_tokens,
        reused=False,
        sequence_length=int(cfg.sequence_length),
    )
