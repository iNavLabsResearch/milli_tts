"""A tiny generic registry (Registry pattern).

Used by the model factory so new architectures can register themselves by name
without the factory having to import every implementation eagerly.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterable, TypeVar

_T = TypeVar("_T")


class Registry(Generic[_T]):
    def __init__(self, name: str) -> None:
        self._name = name
        self._items: Dict[str, Callable[..., _T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
        def deco(fn: Callable[..., _T]) -> Callable[..., _T]:
            if key in self._items:
                raise KeyError(f"'{key}' already registered in {self._name}")
            self._items[key] = fn
            return fn
        return deco

    def get(self, key: str) -> Callable[..., _T]:
        if key not in self._items:
            raise KeyError(
                f"'{key}' not found in registry '{self._name}'. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[key]

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._items
