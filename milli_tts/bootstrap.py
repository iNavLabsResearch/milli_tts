"""Startup bootstrap: load config, resolve secrets, seed, make dirs.

Call :func:`bootstrap` once at the top of any entrypoint (train.py /
inference.py / Colab). It centralizes the boring-but-critical startup steps so
the scripts stay tiny.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from milli_tts.core.logger import get_logger
from milli_tts.core.static_memory_cache import StaticMemoryCache
from milli_tts.utils.seed import seed_everything

log = get_logger("bootstrap")


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (if present)."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def bootstrap(config_path: Optional[str] = None) -> StaticMemoryCache:
    """Initialize the process and return the warmed cache singleton."""
    _load_dotenv()
    cache = StaticMemoryCache.load(config_path)
    cfg = cache.config()

    # propagate secrets into the standard env vars libraries expect
    if cfg.huggingface.token:
        os.environ.setdefault("HF_TOKEN", cfg.huggingface.token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", cfg.huggingface.token)
    if cfg.wandb.api_key:
        os.environ.setdefault("WANDB_API_KEY", cfg.wandb.api_key)

    # create all working directories
    for d in cfg.paths.all():
        os.makedirs(d, exist_ok=True)

    seed_everything(cfg.project.seed)
    log.info("Bootstrapped '%s' (seed=%d). Config: %s",
             cfg.project.name, cfg.project.seed, cache.config_path())
    return cache
