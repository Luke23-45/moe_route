from __future__ import annotations

from torch.utils.data import DataLoader

from moe_route.data.corpus import build_corpus
from moe_route.data.packing import PackedTokenDataset
from moe_route.tokenization.tokenizers import TextTokenizer


def build_dataloader(cfg, tokenizer: TextTokenizer) -> DataLoader:
    corpus = build_corpus(cfg)
    dataset = PackedTokenDataset(corpus.texts(), tokenizer, int(cfg.sequence_length))
    kwargs = {
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(cfg.shuffle),
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
        "persistent_workers": bool(cfg.persistent_workers) if int(cfg.num_workers) > 0 else False,
    }
    if cfg.prefetch_factor is not None and int(cfg.num_workers) > 0:
        kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return DataLoader(dataset, **kwargs)

