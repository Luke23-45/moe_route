from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler

from moe_route.data.corpus import build_corpus
from moe_route.data.packing import PackedTokenDataset, TensorPackedTokenDataset
from moe_route.data.prepare import prepare_data
from moe_route.tokenization.tokenizers import TextTokenizer


def build_dataset(cfg, tokenizer: TextTokenizer, prepare: bool = True):
    if bool(cfg.get("cache_tokenized", False)):
        prepared = prepare_data(cfg, tokenizer, show_progress=prepare, build_missing=prepare)
        samples = torch.load(prepared.samples_path, map_location="cpu", weights_only=True)
        return TensorPackedTokenDataset(samples)
    corpus = build_corpus(cfg)
    return PackedTokenDataset(corpus.texts(), tokenizer, int(cfg.sequence_length))


def build_dataloader(
    cfg,
    tokenizer: TextTokenizer,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    prepare: bool = True,
) -> DataLoader:
    dataset = build_dataset(cfg, tokenizer, prepare=prepare)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=bool(cfg.shuffle))
        if distributed
        else None
    )
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(cfg.shuffle) and sampler is None,
        "sampler": sampler,
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
        "persistent_workers": bool(cfg.persistent_workers) if int(cfg.num_workers) > 0 else False,
    }
    if cfg.prefetch_factor is not None and int(cfg.num_workers) > 0:
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(dataset, **kwargs)
