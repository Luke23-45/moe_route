from __future__ import annotations

import torch

from moe_route.data.packing import PackedTokenDataset, TensorPackedTokenDataset
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
