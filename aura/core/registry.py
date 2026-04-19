"""Name-keyed collection of LangChain BaseTool instances used by the agent loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from aura.core.loop import ToolStep


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool]) -> None:
        by_name: dict[str, BaseTool] = {}
        for t in tools:
            if t.name in by_name:
                raise ValueError(f"duplicate tool name: {t.name!r}")
            by_name[t.name] = t
        self._by_name: dict[str, BaseTool] = by_name

    def __getitem__(self, name: str) -> BaseTool:
        return self._by_name[name]

    def get(self, name: str) -> BaseTool | None:
        return self._by_name.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name)

    def tools(self) -> list[BaseTool]:
        """Return the ordered list of tools; pass directly to ``model.bind_tools``."""
        return list(self._by_name.values())

    @staticmethod
    def partition_batches(steps: list[ToolStep]) -> list[list[ToolStep]]:
        """将 steps 按并发安全性分批（保序，不重排）。

        1. 连续的 is_concurrency_safe 且 decision=None 的 step 合并成一个并行 batch，
           批内用 gather 一次并发执行并保序拿回结果。
        2. 非 safe 或已被 pre_tool 短路（decision 非 None）的 step 单独成 batch。
        3. 维持原 tool_call 顺序 —— 并发只发生在 batch 内，不跨 batch。
        """
        batches: list[list[ToolStep]] = []
        current: list[ToolStep] = []
        for step in steps:
            tool = step.tool
            safe = (
                tool is not None
                and (tool.metadata or {}).get("is_concurrency_safe", False)
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
