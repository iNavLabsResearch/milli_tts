"""StaticMemoryCache — the single source of truth for runtime configuration.

Design patterns
---------------
* **Singleton**: exactly one cache instance exists per process. Every module
  in the codebase reads config through ``StaticMemoryCache.config()`` instead
  of threading a config object through every constructor.
* **Lazy initialization**: the JSON file is parsed (and ``ENV:`` secrets
  resolved) on first access, then memoized.
* **Registry / object store**: besides config, the cache doubles as a process
  wide store for heavy, expensive-to-build singletons (the Mimi codec, the
  tokenizer, the resolved torch device) so they are constructed at most once.

Thread-safety is provided via a re-entrant lock so the cache can be safely
warmed from a DataLoader worker.

Usage
-----
    from milli_tts import StaticMemoryCache

    StaticMemoryCache.load("config.json")        # call once at startup
    cfg = StaticMemoryCache.config()             # anywhere afterwards
    device = StaticMemoryCache.device()
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from milli_tts.core.config import AppConfig

_DEFAULT_CONFIG_NAME = "config.json"


class StaticMemoryCache:
    """Process-wide singleton holding config + shared heavy objects."""

    _instance: Optional["StaticMemoryCache"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:  # pragma: no cover - use load()/instance()
        if StaticMemoryCache._instance is not None:
            raise RuntimeError(
                "StaticMemoryCache is a singleton; use StaticMemoryCache.load()."
            )
        self._config: Optional[AppConfig] = None
        self._config_path: Optional[Path] = None
        self._store: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Singleton accessors
    # ------------------------------------------------------------------ #
    @classmethod
    def instance(cls) -> "StaticMemoryCache":
        """Return the live singleton, creating an empty one if needed."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                cls._instance._config = None
                cls._instance._config_path = None
                cls._instance._store = {}
            return cls._instance

    @classmethod
    def load(cls, config_path: str | os.PathLike | None = None,
             *, force: bool = False) -> "StaticMemoryCache":
        """Load and memoize ``config.json``.

        Args:
            config_path: path to the JSON config. If ``None`` we search the
                current working directory and then the repository root.
            force: re-read the file even if already loaded.
        """
        inst = cls.instance()
        with cls._lock:
            if inst._config is not None and not force:
                return inst
            path = cls._resolve_path(config_path)
            with open(path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = json.load(fh)
            inst._config = AppConfig.from_dict(raw)
            inst._config_path = path
            return inst

    # ------------------------------------------------------------------ #
    # Config access
    # ------------------------------------------------------------------ #
    @classmethod
    def config(cls) -> AppConfig:
        """Return the parsed :class:`AppConfig`, loading defaults if needed."""
        inst = cls.instance()
        if inst._config is None:
            cls.load()
        assert inst._config is not None
        return inst._config

    @classmethod
    def config_path(cls) -> Optional[Path]:
        return cls.instance()._config_path

    # ------------------------------------------------------------------ #
    # Generic object store (lazy singletons)
    # ------------------------------------------------------------------ #
    @classmethod
    def get_or_create(cls, key: str, factory: Callable[[], Any]) -> Any:
        """Return cached object for ``key`` or build it once via ``factory``."""
        inst = cls.instance()
        with cls._lock:
            if key not in inst._store:
                inst._store[key] = factory()
            return inst._store[key]

    @classmethod
    def put(cls, key: str, value: Any) -> None:
        with cls._lock:
            cls.instance()._store[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return cls.instance()._store.get(key, default)

    @classmethod
    def clear_store(cls) -> None:
        with cls._lock:
            cls.instance()._store.clear()

    # ------------------------------------------------------------------ #
    # Convenience: resolved torch device
    # ------------------------------------------------------------------ #
    @classmethod
    def device(cls, requested: Optional[str] = None):
        """Resolve and memoize the torch device.

        ``requested`` may be ``"auto"``, ``"cuda"``, ``"cpu"`` or ``"mps"``.
        ``auto`` prefers CUDA, then Apple MPS, then CPU.
        """
        import torch  # local import keeps config import torch-free

        def _build():
            req = requested or "auto"
            if req == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(req)

        return cls.get_or_create(f"device::{requested or 'auto'}", _build)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_path(config_path: str | os.PathLike | None) -> Path:
        if config_path is not None:
            p = Path(config_path).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {p}")
            return p
        # search cwd, then the package's repository root (two parents up).
        candidates = [
            Path.cwd() / _DEFAULT_CONFIG_NAME,
            Path(__file__).resolve().parents[2] / _DEFAULT_CONFIG_NAME,
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"Could not locate {_DEFAULT_CONFIG_NAME}. Looked in: "
            + ", ".join(str(c) for c in candidates)
        )

    # ------------------------------------------------------------------ #
    # Testing helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _reset(cls) -> None:
        """Drop the singleton entirely (used by tests)."""
        with cls._lock:
            cls._instance = None
