"""Distributed training helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class DistributedState:
    """Process rank and device state."""

    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(device_preference: str) -> DistributedState:
    """Initialize torch.distributed from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif device_preference == "cuda":
        device = torch.device("cpu")
    else:
        device = torch.device(device_preference)

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return DistributedState(distributed, rank, local_rank, world_size, device)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(
    model: nn.Module, state: DistributedState, find_unused_parameters: bool = False
) -> nn.Module:
    """Wrap a model in DistributedDataParallel when needed."""
    if not state.distributed:
        return model
    if state.device.type == "cuda":
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[state.local_rank],
            output_device=state.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return nn.parallel.DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )


def reduce_metrics(metrics: dict[str, torch.Tensor], state: DistributedState) -> dict[str, torch.Tensor]:
    """Average scalar metrics across processes."""
    if not state.distributed:
        return metrics
    reduced: dict[str, torch.Tensor] = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor / state.world_size
    return reduced

