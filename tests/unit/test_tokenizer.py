from __future__ import annotations

from moe_route.tokenization.tokenizers import ByteTokenizer


def test_byte_tokenizer_reserves_special_token_ids() -> None:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("A", add_special_tokens=True)
    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert ids[1] >= 3
    assert tokenizer.decode(ids) == "A"
