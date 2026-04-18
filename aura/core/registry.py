"""Name-keyed collection of AuraTool instances used by the agent loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from aura.core.history import tool_schema_for
from aura.tools.base import AuraTool

if TYPE_CHECKING:
    from aura.core.loop import ToolStep


class ToolRegistry:
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
        return [tool_schema_for(t) for t in self._by_name.values()]

    @staticmethod
    def partition_batches(steps: list[ToolStep]) -> list[list[ToolStep]]:
        batches: list[list[ToolStep]] = []
        current: list[ToolStep] = []
        for step in steps:
            tool = step.tool
            safe = (
                tool is not None
                and tool.is_concurrency_safe
                and step.decision is None
            )
            if safe:
                current.append(step)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([step])
        if current:
            batches.append(current)
        return batches
