"""Minimal DistributedDataParallel (DDP) helpers.

Supports both launch styles with the *same* Trainer code:

* ``torchrun --nproc_per_node=N train.py`` — reads ``RANK`` / ``WORLD_SIZE`` /
  ``LOCAL_RANK`` from the environment, and
* a plain ``python train.py`` that self-spawns one process per visible GPU (see
  ``train.py``). This is what Kaggle's **2×T4** notebooks use — you just run the
  script and both GPUs are used.

Everything degrades to a no-op single-process path when ``WORLD_SIZE <= 1``, so
the identical code runs on CPU / single-GPU / multi-GPU / a multi-node cluster.
Rank/size are read from env vars (not the live process group) so the values are
correct inside forked DataLoader workers too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from milli_tts.core.logger import get_logger

log = get_logger("training.distributed")


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")


def dist_info() -> DistInfo:
    """Read rank/world/local_rank from the environment (works in workers)."""
    return DistInfo(
        rank=int(os.environ.get("RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def setup_distributed() -> DistInfo:
    """Pin the GPU and (if WORLD_SIZE>1) initialize the process group. Idempotent."""
    info = dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    if info.is_distributed and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=info.rank,
                                world_size=info.world_size)
        log.info("DDP initialized: rank %d/%d (local_rank %d, backend=%s)",
                 info.rank, info.world_size, info.local_rank, backend)
    return info


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
