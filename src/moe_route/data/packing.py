from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from moe_route.tokenization.tokenizers import TextTokenizer


def tokenize_texts(texts: Iterable[str], tokenizer: TextTokenizer) -> torch.Tensor:
    token_buffer: list[int] = []
    for text in texts:
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=True))
    if not token_buffer:
        raise ValueError("Corpus produced no tokens.")
    return torch.tensor(token_buffer, dtype=torch.long)


class PackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, texts: Iterable[str], tokenizer: TextTokenizer, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        tokens = tokenize_texts(texts, tokenizer)
        self.samples = pack_tokens(tokens, sequence_length)

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return sample[:-1], sample[1:]


class TensorPackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: torch.Tensor) -> None:
        if samples.ndim != 2 or samples.shape[1] < 3:
            raise ValueError("Packed samples must have shape [num_samples, sequence_length + 1].")
        self.samples = samples.long().contiguous()

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return sample[:-1], sample[1:]


def pack_tokens(tokens: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    stride = sequence_length + 1
    usable = (tokens.numel() // stride) * stride
    if usable < stride:
        raise ValueError("Not enough tokens to create one packed sample.")
    return tokens[:usable].view(-1, stride).contiguous()
