from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler

from moe_route.data.corpus import build_corpus
from moe_route.data.packing import MemoryMappedPackedDataset, PackedTokenDataset
from moe_route.data.prepare import prepare_data
from moe_route.tokenization.tokenizers import TextTokenizer


def build_dataset(cfg, tokenizer: TextTokenizer, prepare: bool = True):
    if bool(cfg.get("cache_tokenized", False)):
        prepared = prepare_data(cfg, tokenizer, show_progress=prepare, build_missing=prepare)
        return MemoryMappedPackedDataset(
            path=prepared.samples_path,
            num_samples=prepared.num_samples,
            sequence_length=prepared.sequence_length,
        )
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
    if bool(cfg.get("dynamic_stress", False)):
        from moe_route.data.samplers import DynamicStressSampler
        sampler = DynamicStressSampler(
            dataset=dataset,
            batch_size=int(cfg.batch_size),
            rank=rank,
            world_size=world_size,
            seed=int(cfg.get("seed", 1337)),
            normal_batches=int(cfg.get("normal_batches", 30)),
            burst_batches=int(cfg.get("burst_batches", 5)),
            tokenizer=tokenizer,
        )
    else:
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
