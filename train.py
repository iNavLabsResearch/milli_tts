#!/usr/bin/env python
"""milli_tts training entrypoint.

Usage:

    python train.py                       # uses ./config.json; auto multi-GPU
    python train.py --config config.json
    python train.py --max-steps 5000      # quick override
    python train.py --no-ddp              # force single-GPU even with 2 GPUs

Multi-GPU (e.g. Kaggle 2×T4): just run `python train.py`. If more than one CUDA
device is visible it self-spawns one DDP worker per GPU — no `torchrun` needed.
Launching via `torchrun --nproc_per_node=N train.py` is also supported (and is
the path for multi-node clusters); the script detects WORLD_SIZE and skips the
self-spawn. Everything else is driven by ``config.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `milli_tts` importable when run directly (e.g. `python train.py`) without
# `pip install -e .` — add the repo root to sys.path BEFORE importing the package.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milli_tts.core.logger import get_logger  # noqa: E402

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train milli_tts")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override training.max_steps")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override training.batch_size")
    p.add_argument("--resume", default=None,
                   help="Override training.resume_from (latest/best/path/none)")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    p.add_argument("--no-ddp", action="store_true",
                   help="Force single-GPU even when multiple GPUs are visible")
    return p.parse_args()


def _run(args: argparse.Namespace) -> None:
    """Bootstrap + train inside the current (possibly spawned) process."""
    from milli_tts.bootstrap import bootstrap
    from milli_tts.core.static_memory_cache import StaticMemoryCache
    from milli_tts.training import Trainer
    from milli_tts.training.distributed import cleanup_distributed

    bootstrap(args.config)
    cfg = StaticMemoryCache.config()

    # apply CLI overrides onto the (frozen) config sections
    if args.max_steps is not None:
        object.__setattr__(cfg.training, "max_steps", args.max_steps)
    if args.batch_size is not None:
        object.__setattr__(cfg.training, "batch_size", args.batch_size)
    if args.resume is not None:
        object.__setattr__(cfg.training, "resume_from", args.resume)
    if args.no_wandb:
        object.__setattr__(cfg.wandb, "enabled", False)

    try:
        Trainer(cfg).train()
    finally:
        cleanup_distributed()


def _spawn_worker(local_rank: int, world_size: int,
                  args: argparse.Namespace) -> None:
    """Entry for each spawned DDP process: publish rank env, then train."""
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    _run(args)
    # Clean hard-exit per worker (see note in __main__): skip Py_Finalize, where
    # HF-streaming's background C thread can emit a cosmetic shutdown crash.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main() -> None:
    args = parse_args()

    # Already under a launcher (torchrun / multi-node) — just run this rank.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run(args)
        return

    import torch

    ngpu = torch.cuda.device_count()
    if ngpu > 1 and not args.no_ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        import torch.multiprocessing as mp

        log.info("Detected %d GPUs — spawning %d DDP workers (one per GPU).",
                 ngpu, ngpu)
        mp.spawn(_spawn_worker, args=(ngpu, args), nprocs=ngpu, join=True)
    else:
        _run(args)


if __name__ == "__main__":
    main()
    # Clean hard-exit: training (incl. W&B finish + final checkpoint) is done.
    # Skips Py_Finalize, where HF streaming's background C thread can crash with
    # a cosmetic "PyGILState_Release … finalizing".
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)