#!/usr/bin/env python
"""milli_tts training entrypoint.

Colab / CLI usage:

    python train.py                       # uses ./config.json
    python train.py --config config.json
    python train.py --max-steps 5000      # quick override

Everything else (dataset, model, optimizer, W&B, checkpoints) is driven by
``config.json`` through the StaticMemoryCache singleton — no other flags needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `milli_tts` importable when run directly (e.g. `python train.py`) without
# `pip install -e .` — add the repo root to sys.path BEFORE importing the package.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from milli_tts.bootstrap import bootstrap  # noqa: E402
from milli_tts.core.logger import get_logger  # noqa: E402
from milli_tts.core.static_memory_cache import StaticMemoryCache  # noqa: E402
from milli_tts.training import Trainer  # noqa: E402

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
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    Trainer(cfg).train()


if __name__ == "__main__":
    import os
    import sys

    main()
    # Clean hard-exit: training (incl. W&B finish + final checkpoint) is done.
    # Skips Py_Finalize, where HF streaming's background C thread can crash with
    # a cosmetic "PyGILState_Release … finalizing".
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
