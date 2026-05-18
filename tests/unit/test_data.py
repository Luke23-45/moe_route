from __future__ import annotations

import torch

from moe_route.data.packing import (
    MemoryMappedPackedDataset,
    PackedTokenDataset,
    TensorPackedTokenDataset,
)
from moe_route.tokenization.tokenizers import ByteTokenizer


def test_packed_dataset_produces_shifted_pairs() -> None:
    ds = PackedTokenDataset(["hello world " * 20], ByteTokenizer(), sequence_length=8)
    x, y = ds[0]
    assert x.shape == y.shape
    assert x.numel() == 8


def test_cached_packed_dataset_roundtrip(tmp_path) -> None:
    _ = tmp_path
    samples = torch.arange(18).view(2, 9)
    ds = TensorPackedTokenDataset(samples)
    x, y = ds[0]
    assert len(ds) == 2
    assert x.tolist() == list(range(8))
    assert y.tolist() == list(range(1, 9))


def test_memory_mapped_packed_dataset_reads_without_full_load(tmp_path) -> None:
    path = tmp_path / "packed.bin"
    samples = torch.arange(18, dtype=torch.int32).view(2, 9)
    samples.numpy().tofile(path)
    ds = MemoryMappedPackedDataset(path=path, num_samples=2, sequence_length=8)
    x, y = ds[1]
    assert len(ds) == 2
    assert x.tolist() == list(range(9, 17))
    assert y.tolist() == list(range(10, 18))
