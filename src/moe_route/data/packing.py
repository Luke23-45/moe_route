from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from moe_route.tokenization.tokenizers import TextTokenizer


class PackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, texts: Iterable[str], tokenizer: TextTokenizer, sequence_length: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")
        self.sequence_length = sequence_length
        token_buffer: list[int] = []
        for text in texts:
            token_buffer.extend(tokenizer.encode(text, add_special_tokens=True))

        stride = sequence_length + 1
        self.samples: list[torch.Tensor] = []
        for start in range(0, max(len(token_buffer) - stride + 1, 0), stride):
            chunk = token_buffer[start : start + stride]
            if len(chunk) == stride:
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
        if not self.samples:
            raise ValueError("Not enough tokens to create one packed sample.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        return sample[:-1], sample[1:]

