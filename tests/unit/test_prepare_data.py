from __future__ import annotations

from omegaconf import OmegaConf

from moe_route.data.prepare import prepare_data
from moe_route.tokenization.tokenizers import ByteTokenizer


def test_prepare_data_reuses_manifest(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "fixture",
            "adapter": "tiny_text",
            "root_dir": str(tmp_path / "data"),
            "raw_dir": str(tmp_path / "data" / "raw"),
            "prepared_dir": str(tmp_path / "data" / "prepared"),
            "cache_dir": str(tmp_path / "data" / "cache"),
            "cache_tokenized": True,
            "rebuild_cache": False,
            "dataset_name": None,
            "dataset_config": None,
            "split": None,
            "streaming": False,
            "text_field": "text",
            "local_texts": ["research code " * 20],
            "max_samples": 2,
            "sequence_length": 8,
            "prepare_chunk_sequences": 4,
        }
    )
    tokenizer = ByteTokenizer()
    first = prepare_data(cfg, tokenizer, show_progress=False)
    second = prepare_data(cfg, tokenizer, show_progress=False)
    assert first.samples_path == second.samples_path
    assert first.samples_path.suffix == ".bin"
    assert first.num_samples > 0
    assert not first.reused
    assert second.reused


def test_prepare_data_creates_manual_holdout_shards(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "name": "fixture_holdout",
            "adapter": "tiny_text",
            "root_dir": str(tmp_path / "data"),
            "raw_dir": str(tmp_path / "data" / "raw"),
            "prepared_dir": str(tmp_path / "data" / "prepared"),
            "cache_dir": str(tmp_path / "data" / "cache"),
            "cache_tokenized": True,
            "rebuild_cache": False,
            "dataset_name": None,
            "dataset_config": None,
            "split": "train",
            "prepared_split": "train",
            "holdout_fraction": 0.3,
            "holdout_seed": 1337,
            "streaming": False,
            "text_field": "text",
            "local_texts": [f"research sample {i} " * 20 for i in range(20)],
            "max_samples": 20,
            "sequence_length": 8,
            "prepare_chunk_sequences": 4,
        }
    )
    tokenizer = ByteTokenizer()
    train_shard = prepare_data(cfg, tokenizer, show_progress=False)
    val_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    val_cfg.prepared_split = "validation"
    val_shard = prepare_data(val_cfg, tokenizer, show_progress=False)

    assert train_shard.samples_path != val_shard.samples_path
    assert train_shard.num_samples > 0
    assert val_shard.num_samples > 0
    assert val_shard.reused
