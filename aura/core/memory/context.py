"""发给模型的 message list 组装 —— 整个代码库里唯一的构造点。

上下文按可变性分四层：
  L1  SystemMessage                  —— 冻结
  L2  <project-memory>               —— eager primary + unconditional rules
  L2b <nested-memory> / <rule>       —— progressive 触发加载，session 内 append-only
  L3  *history                       —— 按 turn 增长

**不变量**：`Context` 自身 progressive 字段
（`_loaded_nested_paths` / `_nested_fragments` / `_matched_rule_paths` /
`_matched_rules`）一旦写入永不移除；`/clear` 与 `/compact` 通过**构造新
Context 实例**实现清空，不原地 reset。

Provider 注入的 section（如 `<todos>`）不计入上述不变量 —— 它们每次 build
从外部 state 读快照，可变性生活在 `LoopState` 侧，Context 仍是纯函数。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from aura.core.memory import project_memory
from aura.core.memory.rules import Rule, RulesBundle
from aura.core.memory.rules import match as match_rules
from aura.core.todos import TodoItem, render_todos_body

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"


@dataclass(frozen=True)
class NestedFragment:
    source: Path
    content: str


class Context:
    """Message-assembly site + session-wide progressive state."""

    def __init__(
        self,
        *,
        cwd: Path,
        system_prompt: str,
        primary_memory: str,
        rules: RulesBundle,
        todos_provider: Callable[[], list[TodoItem]] | None = None,
    ) -> None:
        self._cwd = cwd.resolve()
        self._system_prompt = system_prompt
        self._primary_memory = primary_memory
        self._rules = rules
        self._loaded_nested_paths: set[Path] = set()
        self._nested_fragments: list[NestedFragment] = []
        self._matched_rule_paths: set[Path] = set()
        self._matched_rules: list[Rule] = []
        # Provider snapshot each build (mutability lives in LoopState, not here).
        self._todos_provider = todos_provider

    # ------------------------------------------------------------------
    # Progressive state mutation (append-only within a session)
    # ------------------------------------------------------------------

    def on_tool_touched_path(self, path: Path) -> None:
        try:
            resolved_path = path.resolve()
        except OSError:
            return

        if _is_under(resolved_path, self._cwd):
            self._load_nested_for(resolved_path)

        # User-layer conditional rules with `**/*.py` 可匹配任何绝对路径，
        # 因此 rule match 不受 "path 是否在 cwd 下" 的限制。
        for rule in match_rules(self._rules, resolved_path):
            if rule.source_path in self._matched_rule_paths:
                continue
            self._matched_rule_paths.add(rule.source_path)
            self._matched_rules.append(rule)

    def _load_nested_for(self, resolved_path: Path) -> None:
        """从 `resolved_path.parent` 走到（但不含）`self._cwd`，逐层加载候选文件。"""
        # 若 touched path 就是 cwd（或更浅），其 parent 会跳到 cwd 之外 —— 直接略过。
        start = resolved_path.parent
        if not _is_under(start, self._cwd):
            return
        for intermediate in _intermediate_dirs(start, self._cwd):
            for candidate in (
                intermediate / _AURA_MD,
                intermediate / _AURA_DIR / _AURA_MD,
                intermediate / _AURA_LOCAL_MD,
            ):
                try:
                    resolved_candidate = candidate.resolve()
                except OSError:
                    continue
                if resolved_candidate in self._loaded_nested_paths:
                    continue
                if not candidate.is_file():
                    continue
                content = project_memory.read_with_imports(candidate)
                if content is None:
                    continue
                self._loaded_nested_paths.add(resolved_candidate)
                self._nested_fragments.append(
                    NestedFragment(source=resolved_candidate, content=content)
                )

    # ------------------------------------------------------------------
    # Message assembly
    # ------------------------------------------------------------------

    def build(self, history: list[BaseMessage]) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(self._system_prompt)]

        eager = _joined_eager(self._primary_memory, self._rules.unconditional)
        if eager:
            messages.append(
                HumanMessage(f"<project-memory>\n{eager}\n</project-memory>")
            )

        for fragment in self._nested_fragments:
            messages.append(
                HumanMessage(
                    f'<nested-memory path="{fragment.source}">\n'
                    f"{fragment.content}\n"
                    "</nested-memory>"
                )
            )

        for rule in self._matched_rules:
            messages.append(
                HumanMessage(
                    f'<rule src="{rule.source_path}">\n'
                    f"{rule.content}\n"
                    "</rule>"
                )
            )

        # todos 由 provider 按需读取 LoopState 快照；非空才发一条 HumanMessage，
        # 位置固定在 <rule> 段之后、history 之前。
        if self._todos_provider is not None:
            todos = self._todos_provider()
            if todos:
                body = render_todos_body(todos)
                messages.append(HumanMessage(f"<todos>\n{body}\n</todos>"))

        messages.extend(history)
        return messages


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _joined_eager(primary: str, unconditional: list[Rule]) -> str:
    """拼接 primary_memory 与所有 unconditional rules；空片段全部跳过。"""
    pieces: list[str] = []
    if primary:
        pieces.append(primary)
    for rule in unconditional:
        if rule.content:
            pieces.append(rule.content)
    return "\n\n".join(pieces)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _intermediate_dirs(start: Path, cwd: Path) -> list[Path]:
    """从 `start` 向上走到（但不含）`cwd`，返回 outer→inner 顺序的目录序列。

    调用方已保证 `start` 在 `cwd` 之下（见 `_is_under` 检查）。
    """
    chain: list[Path] = []
    current = start
    while current != cwd:
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain
