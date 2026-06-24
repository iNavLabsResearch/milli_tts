"""Centralized logging. Rich console handler when available, else stdlib."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False


def _configure_root(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler: logging.Handler
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False,
                              markup=False, show_time=True)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich optional
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger("milli_tts")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a namespaced logger under the ``milli_tts`` root."""
    _configure_root(level)
    if name is None:
        return logging.getLogger("milli_tts")
    if not name.startswith("milli_tts"):
        name = f"milli_tts.{name}"
    return logging.getLogger(name)
