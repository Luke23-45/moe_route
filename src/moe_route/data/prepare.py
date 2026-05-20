from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
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
        "prepared_split": str(cfg.get("prepared_split", "train")),
        "sequence_length": int(cfg.sequence_length),
        "max_samples": None if cfg.get("max_samples") is None else int(cfg.get("max_samples")),
        "holdout_fraction": float(cfg.get("holdout_fraction", 0.0)),
        "holdout_seed": int(cfg.get("holdout_seed", 1337)),
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


def _clone_cfg(cfg, **updates):
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key, value in updates.items():
        OmegaConf.update(cloned, key, value, force_add=True)
    return cloned


def _target_prepared_split(cfg) -> str:
    return str(cfg.get("prepared_split", "train"))


def _prepare_stress_cache(cfg, tokenizer: TextTokenizer, prepared: PreparedData) -> None:
    if not bool(cfg.get("dynamic_stress", False)):
        return

    from moe_route.data.packing import MemoryMappedPackedDataset
    from moe_route.data.samplers import DynamicStressSampler

    dataset = MemoryMappedPackedDataset(
        path=prepared.samples_path,
        num_samples=prepared.num_samples,
        sequence_length=prepared.sequence_length,
    )
    # Constructing the sampler materializes the shared .stress_indices.pt cache once.
    # Runtime samplers then only shard and shuffle precomputed index pools.
    DynamicStressSampler(
        dataset=dataset,
        batch_size=int(cfg.batch_size),
        rank=0,
        world_size=1,
        seed=int(cfg.get("seed", 1337)),
        normal_batches=int(cfg.get("normal_batches", 30)),
        burst_batches=int(cfg.get("burst_batches", 5)),
        dialogue_threshold=int(cfg.get("dialogue_threshold", 4)),
        punct_threshold=int(cfg.get("punct_threshold", 11)),
        tokenizer=tokenizer,
    )


def _uses_manual_holdout(cfg) -> bool:
    return float(cfg.get("holdout_fraction", 0.0)) > 0.0 and str(cfg.get("split", "")) == "train"


def _is_validation_doc(index: int, holdout_seed: int, holdout_fraction: float) -> bool:
    mixed = (index * 2654435761 + holdout_seed) & 0xFFFFFFFF
    return mixed < int(holdout_fraction * (1 << 32))


def _prepare_single_shard(
    cfg,
    tokenizer: TextTokenizer,
    *,
    samples_path: Path,
    manifest_path: Path,
    show_progress: bool,
) -> PreparedData:
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
        desc=f"prepare:{cfg.name}:{_target_prepared_split(cfg)}",
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
        raise ValueError(
            f"Data source {cfg.name} produced too few tokens to create one packed sample for "
            f"prepared_split={_target_prepared_split(cfg)}."
        )

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


def _prepare_holdout_shards(cfg, tokenizer: TextTokenizer, *, show_progress: bool) -> dict[str, PreparedData]:
    train_cfg = _clone_cfg(cfg, prepared_split="train")
    val_cfg = _clone_cfg(cfg, prepared_split="validation")
    train_samples_path, train_manifest_path = prepared_dataset_paths(train_cfg, tokenizer)
    val_samples_path, val_manifest_path = prepared_dataset_paths(val_cfg, tokenizer)

    width = int(cfg.sequence_length) + 1
    flush_threshold_sequences = int(cfg.get("prepare_chunk_sequences", 8192))
    flush_threshold_tokens = flush_threshold_sequences * width
    max_samples = cfg.get("max_samples")
    total_docs = None if max_samples is None else int(max_samples)
    holdout_fraction = float(cfg.get("holdout_fraction", 0.0))
    holdout_seed = int(cfg.get("holdout_seed", 1337))

    train_buffer: list[int] = []
    val_buffer: list[int] = []
    train_num_samples = 0
    train_num_tokens = 0
    val_num_samples = 0
    val_num_tokens = 0
    train_docs = 0
    val_docs = 0

    iterator = tqdm(
        build_corpus(cfg).texts(),
        total=total_docs,
        desc=f"prepare:{cfg.name}:holdout",
        unit="docs",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    with train_samples_path.open("wb") as train_out, val_samples_path.open("wb") as val_out:
        for idx, text in enumerate(iterator):
            target_buffer = val_buffer if _is_validation_doc(idx, holdout_seed, holdout_fraction) else train_buffer
            if target_buffer is val_buffer:
                val_docs += 1
            else:
                train_docs += 1
            target_buffer.extend(tokenizer.encode(text, add_special_tokens=True))
            if len(train_buffer) >= flush_threshold_tokens:
                written_samples, written_tokens = _flush_buffer(train_out, train_buffer, width)
                train_num_samples += written_samples
                train_num_tokens += written_tokens
            if len(val_buffer) >= flush_threshold_tokens:
                written_samples, written_tokens = _flush_buffer(val_out, val_buffer, width)
                val_num_samples += written_samples
                val_num_tokens += written_tokens
            if show_progress:
                iterator.set_postfix(
                    train_samples=train_num_samples,
                    val_samples=val_num_samples,
                    train_docs=train_docs,
                    val_docs=val_docs,
                    refresh=False,
                )

        written_samples, written_tokens = _flush_buffer(train_out, train_buffer, width)
        train_num_samples += written_samples
        train_num_tokens += written_tokens
        written_samples, written_tokens = _flush_buffer(val_out, val_buffer, width)
        val_num_samples += written_samples
        val_num_tokens += written_tokens

    if train_num_samples == 0 or val_num_samples == 0:
        for path in (train_samples_path, val_samples_path):
            if path.exists():
                path.unlink()
        raise ValueError(
            f"Manual holdout preparation for {cfg.name} produced an empty shard "
            f"(train_samples={train_num_samples}, validation_samples={val_num_samples}). "
            "Increase max_samples, reduce sequence_length, or lower holdout_fraction."
        )

    train_payload = _manifest_payload(train_cfg, tokenizer, num_samples=train_num_samples, num_tokens=train_num_tokens)
    val_payload = _manifest_payload(val_cfg, tokenizer, num_samples=val_num_samples, num_tokens=val_num_tokens)
    write_manifest(train_manifest_path, train_payload)
    write_manifest(val_manifest_path, val_payload)
    return {
        "train": PreparedData(
            samples_path=train_samples_path,
            manifest_path=train_manifest_path,
            num_samples=train_num_samples,
            num_tokens=train_num_tokens,
            reused=False,
            sequence_length=int(cfg.sequence_length),
        ),
        "validation": PreparedData(
            samples_path=val_samples_path,
            manifest_path=val_manifest_path,
            num_samples=val_num_samples,
            num_tokens=val_num_tokens,
            reused=False,
            sequence_length=int(cfg.sequence_length),
        ),
    }


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
        prepared = PreparedData(
            samples_path=samples_path,
            manifest_path=manifest_path,
            num_samples=int(manifest["num_samples"]),
            num_tokens=int(manifest["num_tokens"]),
            reused=True,
            sequence_length=int(manifest["sequence_length"]),
        )
        if build_missing:
            _prepare_stress_cache(cfg, tokenizer, prepared)
        return prepared
    if not build_missing:
        raise FileNotFoundError(
            f"Prepared data for {cfg.name} is missing or stale at {samples_path}. "
            "Run scripts/prepare_data.py or enable trainer.prepare_data=true."
        )
    if _uses_manual_holdout(cfg):
        shards = _prepare_holdout_shards(cfg, tokenizer, show_progress=show_progress)
        prepared = shards[_target_prepared_split(cfg)]
        _prepare_stress_cache(cfg, tokenizer, prepared)
        return prepared

    prepared = _prepare_single_shard(
        cfg,
        tokenizer,
        samples_path=samples_path,
        manifest_path=manifest_path,
        show_progress=show_progress,
    )
    _prepare_stress_cache(cfg, tokenizer, prepared)
    return prepared
