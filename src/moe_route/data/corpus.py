from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusAdapter(ABC):
    name: str

    @abstractmethod
    def texts(self) -> Iterable[str]:
        """Yield normalized text records for tokenization."""


@dataclass(frozen=True)
class TinyTextCorpus(CorpusAdapter):
    local_texts: tuple[str, ...]
    max_samples: int | None = None

    def texts(self) -> Iterator[str]:
        if not self.local_texts:
            raise ValueError("TinyTextCorpus requires at least one local text.")
        emitted = 0
        while self.max_samples is None or emitted < self.max_samples:
            yield self.local_texts[emitted % len(self.local_texts)]
            emitted += 1


@dataclass(frozen=True)
class HuggingFaceCorpus(CorpusAdapter):
    dataset_name: str
    dataset_config: str | None
    split: str
    text_field: str = "text"
    streaming: bool = False
    cache_dir: str | None = None
    max_samples: int | None = None

    def texts(self) -> Iterator[str]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=self.streaming,
            cache_dir=self.cache_dir,
        )
        for idx, row in enumerate(ds):
            if self.max_samples is not None and idx >= self.max_samples:
                break
            text = row.get(self.text_field)
            if isinstance(text, str) and text:
                yield text


def build_corpus(cfg) -> CorpusAdapter:
    if cfg.adapter == "tiny_text":
        return TinyTextCorpus(
            name=cfg.name,
            local_texts=tuple(cfg.local_texts),
            max_samples=None if cfg.max_samples is None else int(cfg.max_samples),
        )
    if cfg.adapter == "huggingface":
        return HuggingFaceCorpus(
            name=cfg.name,
            dataset_name=cfg.dataset_name,
            dataset_config=None if cfg.dataset_config is None else str(cfg.dataset_config),
            split=cfg.split,
            text_field=cfg.text_field,
            streaming=bool(cfg.streaming),
            cache_dir=None if cfg.cache_dir is None else str(cfg.cache_dir),
            max_samples=None if cfg.max_samples is None else int(cfg.max_samples),
        )
    raise ValueError(f"Unknown corpus adapter: {cfg.adapter}")
