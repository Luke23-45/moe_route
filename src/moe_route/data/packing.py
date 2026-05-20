from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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


class MemoryMappedPackedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        num_samples: int,
        sequence_length: int,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        self.path = str(path)
        self.num_samples = int(num_samples)
        self.sequence_length = int(sequence_length)
        self.width = self.sequence_length + 1
        self.dtype = dtype
        self.samples: torch.Tensor | None = None

    def _ensure_open(self) -> torch.Tensor:
        if self.samples is None:
            size = self.num_samples * self.width
            self.samples = torch.from_file(self.path, shared=False, size=size, dtype=self.dtype).view(
                self.num_samples, self.width
            )
        return self.samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._ensure_open()[index].long()
        return sample[:-1], sample[1:]

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["samples"] = None
        return state


def pack_tokens(tokens: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Pack a flat token stream into fixed-width training samples.

    WARNING (CROSS-DOCUMENT CONTAMINATION):
    This function concatenates ALL documents into a single token stream and slices
    into fixed-width chunks. A single chunk may span MULTIPLE document boundaries
    (e.g., [...EOS, BOS, ...]). With causal attention (is_causal=True), tokens from
    document N+1 can attend to document N's tokens within the same packed sample.

    Implications for research:
    1. The model receives "free" context from unrelated documents, inflating
       apparent perplexity (easier predictions at document boundaries).
    2. The loss includes cross-document boundary predictions (predicting next doc's
       BOS given previous doc's EOS), which are inherently noisy/unpredictable.
    3. This is the standard GPT-2 approach and is acceptable for controlled
       comparisons where ALL experiments use the same packing.

    For research-grade accuracy, consider implementing document-boundary attention
    masking or packing documents individually with padding.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    stride = sequence_length + 1
    usable = (tokens.numel() // stride) * stride
    if usable < stride:
        raise ValueError("Not enough tokens to create one packed sample.")
    return tokens[:usable].view(-1, stride).contiguous()
