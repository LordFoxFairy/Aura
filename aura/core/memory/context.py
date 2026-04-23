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
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from aura.core.memory import project_memory
from aura.core.memory.rules import Rule, RulesBundle
from aura.core.memory.rules import match as match_rules
from aura.core.skills.types import Skill
from aura.schemas.todos import TodoItem

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"


@dataclass(frozen=True)
class NestedFragment:
    source: Path
    content: str


@dataclass(frozen=True)
class _ReadRecord:
    mtime: float
    size: int
    partial: bool = False


class Context:
    """Message-assembly site + session-wide progressive state."""

    def __init__(
        self,
        *,
        cwd: Path,
        system_prompt: str,
        primary_memory: str,
        rules: RulesBundle,
        skills: list[Skill] | None = None,
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
        # Skills are static for the session: the available list comes in at
        # construction (rendered as <skills-available> on every build). The
        # invoked list grows append-only via record_skill_invocation and
        # dedups by source_path — mirrors _matched_rules. /clear constructs a
        # new Context instance, which resets _invoked_skills while keeping
        # _skills_available populated (same Skill list passed back in).
        self._skills_available: list[Skill] = list(skills) if skills else []
        self._invoked_skill_paths: set[Path] = set()
        self._invoked_skills: list[Skill] = []
        # Must-read-first invariant (mirrors claude-code FileEditTool.ts:275–287):
        # edit_file must see a prior successful read_file on the same resolved
        # path AND the file must not have changed on disk since. Populated by
        # AgentLoop after a successful read_file call; staleness compared via
        # (mtime, size) — lighter than claude-code's content hash but
        # equivalent in practice for real edits.
        self._read_records: dict[Path, _ReadRecord] = {}
        # Provider snapshot each build (mutability lives in LoopState, not here).
        self._todos_provider = todos_provider

    # ------------------------------------------------------------------
    # Progressive state mutation (append-only within a session)
    # ------------------------------------------------------------------

    def record_skill_invocation(self, skill: Skill) -> None:
        """Append ``skill`` to the invoked list (dedup by ``source_path``).

        Called from ``SkillCommand.handle`` via ``Agent.record_skill_invocation``
        when the user types ``/<skill.name>``. The body is rendered as a
        ``<skill-invoked>`` HumanMessage on the next ``build()`` call; the
        model therefore sees it on the turn *following* invocation.
        """
        if skill.source_path in self._invoked_skill_paths:
            return
        self._invoked_skill_paths.add(skill.source_path)
        self._invoked_skills.append(skill)

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

    def record_read(self, path: Path, *, partial: bool = False) -> None:
        """Mark ``path`` as read in this session (must-read-first invariant).

        ``partial`` — True when the read only returned a slice of the file
        (offset>0 or limit truncated); drives the ``"partial"`` branch of
        ``read_status``. A subsequent full read overwrites the record, so
        partial → fresh recovery is just "read again without offset/limit".

        Silent on resolve/stat failure — the read itself already succeeded,
        so failing to record is non-fatal. The invariant fails closed on the
        read_status side, so a missed record just forces a re-read. Path may
        have disappeared in the race between ainvoke returning and the loop
        calling record_read; benign.
        """
        try:
            resolved = path.resolve()
        except OSError:
            return
        try:
            st = resolved.stat()
        except OSError:
            return
        self._read_records[resolved] = _ReadRecord(
            mtime=st.st_mtime, size=st.st_size, partial=partial,
        )

    def read_status(
        self, path: Path,
    ) -> Literal["never_read", "stale", "partial", "fresh"]:
        """Return the read-state of ``path`` relative to this session.

        - ``"never_read"`` — no prior ``record_read`` (or path unresolvable).
        - ``"stale"``     — recorded, but (mtime, size) changed since, or the
          path has disappeared from disk (recorded-but-now-gone counts as
          stale so the edit surfaces a "has changed" error).
        - ``"partial"``   — recorded, fingerprint still matches, but the
          recorded read only saw a slice (offset>0 or truncated limit).
        - ``"fresh"``     — recorded, fingerprint matches, full read.

        Fail-closed on resolve failure: returning ``"never_read"`` forces a
        re-read, matching claude-code's FileEditTool guard behavior.
        """
        try:
            resolved = path.resolve()
        except OSError:
            return "never_read"
        record = self._read_records.get(resolved)
        if record is None:
            return "never_read"
        try:
            st = resolved.stat()
        except OSError:
            return "stale"
        if (st.st_mtime, st.st_size) != (record.mtime, record.size):
            return "stale"
        if record.partial:
            return "partial"
        return "fresh"

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

        # Skills — system-tier context like rules, placed AFTER rules so
        # rules keep their stable position (better prompt-cache behavior) and
        # BEFORE history so the invoked body influences the model on the
        # turn following invocation. <skills-available> is rendered once
        # (the full catalogue), then one <skill-invoked> per distinct skill.
        #
        # Filter: hide ``disable_model_invocation`` skills (user-only, not
        # the LLM's to auto-pick) and conditional skills still in the lazy
        # bucket (they enter the list only after
        # ``activate_conditional_skills_for_paths`` moves them into the
        # registry — which hands back a refreshed Skill list on the next
        # Context rebuild). When present, ``when_to_use`` is appended as
        # ``[when to use: ...]`` so the model gets explicit triggering
        # guidance alongside the description.
        visible_skills = [
            s for s in self._skills_available
            if not s.disable_model_invocation and not s.is_conditional()
        ]
        if visible_skills:
            available_lines: list[str] = []
            for s in visible_skills:
                line = f"- {s.name}: {s.description}"
                if s.when_to_use:
                    line += f" [when to use: {s.when_to_use}]"
                available_lines.append(line)
            messages.append(
                HumanMessage(
                    "<skills-available>\n"
                    + "\n".join(available_lines)
                    + "\n</skills-available>"
                )
            )
        for skill in self._invoked_skills:
            messages.append(
                HumanMessage(
                    f'<skill-invoked name="{skill.name}">\n'
                    f"{skill.body}\n"
                    "</skill-invoked>"
                )
            )

        # todos 由 provider 按需读取 LoopState 快照；非空才发一条 HumanMessage，
        # 位置固定在 <rule> 段之后、history 之前。
        if self._todos_provider is not None:
            todos = self._todos_provider()
            if todos:
                body = _render_todos_body(todos)
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


def _render_todos_body(todos: list[TodoItem]) -> str:
    """Render todos for the ``<todos>`` HumanMessage body.

    Format is intentional but informal — the model re-reads its own output
    across turns. Completed items get only their content; active items also
    show the present-continuous form so the model can see "what now".
    """
    lines: list[str] = []
    for t in todos:
        if t.status == "completed":
            lines.append(f"- [completed] {t.content}")
        else:
            lines.append(f"- [{t.status}] {t.content} (active: {t.active_form})")
    return "\n".join(lines)


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
