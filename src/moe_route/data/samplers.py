from __future__ import annotations

import logging
from pathlib import Path
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DynamicStressSampler(Sampler[int]):
    """Dynamic Domain-Stress Sampler for MoE Gating Evaluation.

    Partitions dataset samples into Normal, Dialogue-heavy, and Punctuation-dense pools,
    then constructs a DDP-safe periodic sequence of batches (Normal -> Dialogue -> Normal -> Punctuation).
    Ensures identical sample counts across all ranks to prevent distributed training deadlocks.
    Uses file-based caching for instantaneous initialization across worker processes.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        normal_batches: int = 30,
        burst_batches: int = 5,
        dialogue_threshold: int = 4,
        punct_threshold: int = 11,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.normal_batches = normal_batches
        self.burst_batches = burst_batches

        samples_path = getattr(dataset, "path", None)
        cache_path = Path(samples_path).with_suffix(".stress_indices.pt") if samples_path else None

        # 1. Vectorized Classification (with optional caching for memory-mapped datasets)
        if cache_path is not None and cache_path.exists():
            try:
                logger.info(f"Loading cached stress indices from {cache_path}")
                data = torch.load(cache_path, map_location="cpu")
                self.normal_indices = data["normal"]
                self.dialogue_indices = data["dialogue"]
                self.punct_indices = data["punct"]
            except Exception as e:
                logger.warning(f"Failed to load cached indices ({e}). Scanning dataset...")
                self._scan_and_cache(dataset, cache_path, dialogue_threshold, punct_threshold)
        else:
            self._scan_and_cache(dataset, cache_path, dialogue_threshold, punct_threshold)

        if not self.dialogue_indices:
            raise ValueError("Dynamic stress sampling requires at least one dialogue-heavy sample.")
        if not self.punct_indices:
            raise ValueError("Dynamic stress sampling requires at least one punctuation-heavy sample.")
        if not self.normal_indices:
            raise ValueError("Dynamic stress sampling requires at least one normal sample.")

        self._normal_indices = torch.as_tensor(self.normal_indices, dtype=torch.long)
        self._dialogue_indices = torch.as_tensor(self.dialogue_indices, dtype=torch.long)
        self._punct_indices = torch.as_tensor(self.punct_indices, dtype=torch.long)

        # 3. Print stats
        logger.info(
            f"Dataset classification: "
            f"Normal={len(self.normal_indices)}, "
            f"Dialogue={len(self.dialogue_indices)}, "
            f"Punctuation={len(self.punct_indices)}"
        )

        # 4. Partition and Shard Pools for DDP
        # Calculate maximum possible normal batches per rank to keep all ranks perfectly aligned
        self.num_normal_batches_per_rank = len(self.normal_indices) // (world_size * batch_size)
        self.total_normal_samples_per_rank = self.num_normal_batches_per_rank * batch_size

        # Shard pools deterministically
        self.normal_sharded = self._normal_indices[rank::world_size]
        self.dialogue_sharded = self._dialogue_indices[rank::world_size]
        self.punct_sharded = self._punct_indices[rank::world_size]

        # Truncate normal_sharded so all ranks have exactly the same amount of normal data
        self.normal_sharded = self.normal_sharded[:self.total_normal_samples_per_rank]

        if self.dialogue_sharded.numel() == 0:
            raise ValueError("Dynamic stress sampling requires at least one dialogue-heavy sample per rank.")
        if self.punct_sharded.numel() == 0:
            raise ValueError("Dynamic stress sampling requires at least one punctuation-heavy sample per rank.")

    def _scan_and_cache(self, dataset, cache_path: Path | None, dialogue_threshold: int, punct_threshold: int) -> None:
        """Scan token indices and cache the results atomically."""
        num_samples = len(dataset)
        chunk_size = 100000

        quote_counts = torch.zeros(num_samples, dtype=torch.int16)
        punct_counts = torch.zeros(num_samples, dtype=torch.int16)

        samples = None
        if hasattr(dataset, "_ensure_open"):
            samples = dataset._ensure_open()
        elif hasattr(dataset, "samples") and isinstance(dataset.samples, torch.Tensor):
            samples = dataset.samples

        if samples is not None:
            for i in range(0, num_samples, chunk_size):
                end_idx = min(i + chunk_size, num_samples)
                chunk = samples[i:end_idx]
                quote_counts[i:end_idx] = (chunk == 37).sum(dim=1).to(torch.int16)
                punct_mask = (chunk == 49) | (chunk == 47) | (chunk == 66) | (chunk == 36)
                punct_counts[i:end_idx] = punct_mask.sum(dim=1).to(torch.int16)
        else:
            for i in range(num_samples):
                sample, _ = dataset[i]
                quote_counts[i] = int((sample == 37).sum().item())
                punct_mask = (sample == 49) | (sample == 47) | (sample == 66) | (sample == 36)
                punct_counts[i] = int(punct_mask.sum().item())

        # Apply thresholds to create boolean masks
        is_dialogue = quote_counts >= dialogue_threshold
        is_punct = (punct_counts >= punct_threshold) & (~is_dialogue)
        is_normal = ~(is_dialogue | is_punct)

        # Extract indices
        self.dialogue_indices = torch.nonzero(is_dialogue).squeeze(1).tolist()
        self.punct_indices = torch.nonzero(is_punct).squeeze(1).tolist()
        self.normal_indices = torch.nonzero(is_normal).squeeze(1).tolist()

        # Atomic file save to prevent corruption in multi-rank DDP launches
        if cache_path is not None:
            temp_path = cache_path.with_suffix(".tmp")
            torch.save(
                {
                    "normal": self.normal_indices,
                    "dialogue": self.dialogue_indices,
                    "punct": self.punct_indices,
                },
                temp_path,
            )
            try:
                temp_path.replace(cache_path)
            except Exception:
                pass

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Deterministically shuffle the sharded pools for this epoch
        normal = self.normal_sharded[torch.randperm(self.normal_sharded.numel(), generator=g)]
        dialogue = self.dialogue_sharded[torch.randperm(self.dialogue_sharded.numel(), generator=g)]
        punct = self.punct_sharded[torch.randperm(self.punct_sharded.numel(), generator=g)]

        normal_ptr = 0
        dialogue_ptr = 0
        punct_ptr = 0
        batches: list[torch.Tensor] = []

        # Cycle type: 0: normal, 1: dialogue, 2: normal, 3: punctuation
        cycle_type = 0
        batches_left_in_phase = self.normal_batches

        # Yield exact multiples of self.batch_size until normal pool is exhausted
        while normal_ptr + self.batch_size <= normal.numel():
            if cycle_type in (0, 2):  # Normal phase
                batches.append(normal[normal_ptr:normal_ptr + self.batch_size])
                normal_ptr += self.batch_size
                batches_left_in_phase -= 1
                if batches_left_in_phase == 0:
                    if cycle_type == 0:
                        cycle_type = 1
                        batches_left_in_phase = self.burst_batches
                    else:
                        cycle_type = 3
                        batches_left_in_phase = self.burst_batches

            elif cycle_type == 1:  # Dialogue burst phase
                next_ptr = dialogue_ptr + self.batch_size
                if next_ptr <= dialogue.numel():
                    batch = dialogue[dialogue_ptr:next_ptr]
                else:
                    wrap = next_ptr - dialogue.numel()
                    batch = torch.cat((dialogue[dialogue_ptr:], dialogue[:wrap]))
                batches.append(batch)
                dialogue_ptr = next_ptr % dialogue.numel()
                batches_left_in_phase -= 1
                if batches_left_in_phase == 0:
                    cycle_type = 2
                    batches_left_in_phase = self.normal_batches

            elif cycle_type == 3:  # Punctuation burst phase
                next_ptr = punct_ptr + self.batch_size
                if next_ptr <= punct.numel():
                    batch = punct[punct_ptr:next_ptr]
                else:
                    wrap = next_ptr - punct.numel()
                    batch = torch.cat((punct[punct_ptr:], punct[:wrap]))
                batches.append(batch)
                punct_ptr = next_ptr % punct.numel()
                batches_left_in_phase -= 1
                if batches_left_in_phase == 0:
                    cycle_type = 0
                    batches_left_in_phase = self.normal_batches

        if not batches:
            return iter(())
        return iter(torch.cat(batches).tolist())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        # Calculate exactly how many samples will be yielded per rank
        # 1 cycle is (normal_batches * 2 + burst_batches * 2) batches
        cycle_normal_batches = self.normal_batches * 2
        cycle_burst_batches = self.burst_batches * 2
        batches_per_cycle = cycle_normal_batches + cycle_burst_batches

        num_cycles = self.num_normal_batches_per_rank // cycle_normal_batches
        remaining_normal_batches = self.num_normal_batches_per_rank % cycle_normal_batches

        total_batches = num_cycles * batches_per_cycle
        if remaining_normal_batches > 0:
            if remaining_normal_batches <= self.normal_batches:
                total_batches += remaining_normal_batches
            else:
                total_batches += self.normal_batches + self.burst_batches
                total_batches += (remaining_normal_batches - self.normal_batches)

        return total_batches * self.batch_size
