from __future__ import annotations

from moe_route.data.packing import PackedTokenDataset
from moe_route.tokenization.tokenizers import ByteTokenizer


def test_packed_dataset_produces_shifted_pairs() -> None:
    ds = PackedTokenDataset(["hello world " * 20], ByteTokenizer(), sequence_length=8)
    x, y = ds[0]
    assert x.shape == y.shape
    assert x.numel() == 8

