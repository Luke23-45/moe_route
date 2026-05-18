from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(cfg) -> DistributedContext:
    enabled = bool(cfg.distributed.enabled)
    if not enabled:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(False, 0, 0, 1, device)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=str(cfg.distributed.backend), rank=rank, world_size=world_size)
    return DistributedContext(True, rank, local_rank, world_size, device)


def wrap_model(model: nn.Module, ctx: DistributedContext, find_unused_parameters: bool) -> nn.Module:
    if not ctx.enabled:
        return model
    kwargs = {"find_unused_parameters": find_unused_parameters}
    if ctx.device.type == "cuda":
        kwargs["device_ids"] = [ctx.local_rank]
    return DistributedDataParallel(model, **kwargs)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()

