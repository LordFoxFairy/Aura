"""Tool registry — wraps a dict with uniqueness + schema emission + partitioning."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from aura.core.history import tool_schema_for
from aura.tools.base import AuraTool


class ToolRegistry:
    """Immutable, name-keyed collection of AuraTool instances.

    Built once per Agent. Enforces unique names at construction. Emits LangChain
    tool schemas on demand. Future-ready: `partition_batches()` will group
    consecutive concurrency-safe calls for parallel dispatch (MVP is serial).
    """

    def __init__(self, tools: Iterable[AuraTool]) -> None:
        by_name: dict[str, AuraTool] = {}
        for t in tools:
            if t.name in by_name:
                raise ValueError(f"duplicate tool name: {t.name!r}")
            by_name[t.name] = t
        self._by_name: dict[str, AuraTool] = by_name

    def __getitem__(self, name: str) -> AuraTool:
        return self._by_name[name]

    def get(self, name: str) -> AuraTool | None:
        return self._by_name.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self) -> Iterator[AuraTool]:
        return iter(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name)

    def schemas(self) -> list[dict[str, object]]:
        """LangChain tool-binding dicts in registration order."""
        return [tool_schema_for(t) for t in self._by_name.values()]

    # Future: partition_batches(calls) → list[list[call]] for parallel dispatch.
    # Stub omitted for MVP per spec §7 "Concurrent batching: phase 4+".
