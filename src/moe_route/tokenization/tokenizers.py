from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class TextTokenizer(Protocol):
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    vocab_size: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: Iterable[int]) -> str: ...


@dataclass
class ByteTokenizer:
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    vocab_size: int = 256

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [int(b) for b in text.encode("utf-8", errors="replace")]
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        filtered = [i for i in ids if i not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}]
        return bytes([i for i in filtered if 0 <= i <= 255]).decode("utf-8", errors="replace")


class BpeTokenizer:
    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path))
        vocab = self.tokenizer.get_vocab()
        self.pad_token_id = vocab.get("<pad>", 0)
        self.bos_token_id = vocab.get("<bos>", 1)
        self.eos_token_id = vocab.get("<eos>", 2)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids))


def train_bpe_tokenizer(
    texts: Iterable[str],
    output_path: str | Path,
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> BpeTokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens or ["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(output))
    return BpeTokenizer(output)


def build_tokenizer(cfg) -> TextTokenizer:
    kind = cfg.kind
    if kind == "byte":
        return ByteTokenizer(
            pad_token_id=int(cfg.pad_token_id),
            bos_token_id=int(cfg.bos_token_id),
            eos_token_id=int(cfg.eos_token_id),
            vocab_size=int(cfg.vocab_size),
        )
    if kind == "bpe":
        return BpeTokenizer(cfg.path)
    raise ValueError(f"Unknown tokenizer kind: {kind}")
