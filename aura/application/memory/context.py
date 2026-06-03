"""Single message-assembly site; mutability ladder L1 sys / L2 eager / L2b progressive / L3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from aura.application.memory import project_memory
from aura.application.memory.rules import Rule, RulesBundle
from aura.application.memory.rules import match as match_rules
from aura.domain.skill import Skill
from aura.domain.state_values import ReadCarryover
from aura.domain.task import TaskNotification
from aura.domain.todos import TodoItem
from aura.infrastructure.persistence import journal

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"

# Makes project instructions outrank default behavior.
_OVERRIDE_PREAMBLE = (
    "Codebase and user instructions are shown below. Be sure to adhere to "
    "these instructions. IMPORTANT: These instructions OVERRIDE any default "
    "behavior and you MUST follow them exactly as written."
)


@dataclass(frozen=True)
class NestedFragment:
    source: Path
    content: str


@dataclass(frozen=True)
class ReadRecord:
    mtime: float
    size: int
    partial: bool = False


class Context:
    def __init__(
        self,
        *,
        cwd: Path,
        system_prompt: str,
        primary_memory: str,
        rules: RulesBundle,
        skills: list[Skill] | None = None,
        todos_provider: Callable[[], list[TodoItem]] | None = None,
        notifications_drainer: Callable[[], list[TaskNotification]] | None = None,
        carryover: ReadCarryover | None = None,
        model: BaseChatModel | None = None,
    ) -> None:
        self._cwd = cwd.resolve()
        self._system_prompt = system_prompt
        self._primary_memory = primary_memory
        self._rules = rules
        self._loaded_nested_paths: set[Path] = set()
        self._nested_fragments: list[NestedFragment] = []
        self._matched_rule_paths: set[Path] = set()
        self._matched_rules: list[Rule] = []
        # _skills_available is static (constructor-injected); _invoked_skills is
        # append-only via record_skill_invocation, dedup by source_path.
        self._skills_available: list[Skill] = list(skills) if skills else []
        self._invoked_skill_paths: set[Path] = set()
        self._invoked_skills: list[Skill] = []
        # Must-read-first invariant: edit_file requires a prior read_file on the
        # same resolved path AND the file unchanged since. (mtime, size) compared
        # — lighter than content hash, equivalent in practice.
        self._read_records: dict[Path, ReadRecord] = {}
        if carryover is not None:
            # Parent's full reads become non-partial seeds.
            for path, record in carryover.records.items():
                self._read_records[path] = ReadRecord(
                    mtime=record.mtime_at_read,
                    size=record.size_at_read,
                    partial=False,
                )
        # Providers read external snapshots each build — mutability lives elsewhere.
        self._todos_provider = todos_provider
        self._notifications_drainer = notifications_drainer
        # Held only to sniff provider type for cache_control; never invoked through.
        self._model = model

    def fresh(
        self,
        *,
        carryover: ReadCarryover | None = None,
        clear_reads: bool = False,
    ) -> Context:
        """New Context with empty progressive state; `/clear` and `/compact` use this.

        `carryover` seeds `_read_records` from a parent Agent's snapshot (subagent
        spawn). `clear_reads=True` wipes them outright (`/clear`). The two are
        mutually exclusive. Default preserves reads — the file is still on disk
        after a compact.
        """
        if carryover is not None and clear_reads:
            raise ValueError(
                "fresh(): pass either carryover or clear_reads=True, not both",
            )

        new_ctx = Context(
            cwd=self._cwd,
            system_prompt=self._system_prompt,
            primary_memory=self._primary_memory,
            rules=self._rules,
            skills=self._skills_available,
            todos_provider=self._todos_provider,
            notifications_drainer=self._notifications_drainer,
            carryover=carryover,
            model=self._model,
        )
        if clear_reads:
            new_ctx._read_records = {}
        elif carryover is None:
            # Shallow copy so the old instance (alive in journal-replay paths)
            # doesn't retroactively see new records.
            new_ctx._read_records = dict(self._read_records)
        return new_ctx

    @property
    def read_records(self) -> dict[Path, ReadRecord]:
        # Live dict — compact/restore paths need direct mutation; readers must copy.
        return self._read_records

    @property
    def invoked_skills(self) -> list[Skill]:
        return self._invoked_skills

    def bind_read_records(self, records: dict[Path, ReadRecord]) -> None:
        """Replace the read-record map; compact preserves reads across rebuild."""
        self._read_records = records

    def record_skill_invocation(self, skill: Skill) -> None:
        if skill.source_path in self._invoked_skill_paths:
            return
        self._invoked_skill_paths.add(skill.source_path)
        self._invoked_skills.append(skill)

    def _path_in_scope(self, path: Path) -> bool:
        # Resolved through symlinks: symlink-out-of-cwd is "outside".
        try:
            resolved_path = path.resolve()
        except OSError:
            return False
        return _is_under(resolved_path, self._cwd)

    def on_tool_touched_path(self, path: Path) -> None:
        try:
            resolved_path = path.resolve()
        except OSError:
            return

        # Out-of-cwd paths cannot trigger nested-memory or conditional rules.
        if not self._path_in_scope(resolved_path):
            return

        self._load_nested_for(resolved_path)

        for rule in match_rules(self._rules, resolved_path):
            if rule.source_path in self._matched_rule_paths:
                continue
            self._matched_rule_paths.add(rule.source_path)
            self._matched_rules.append(rule)

    def record_read(self, path: Path, *, partial: bool = False) -> None:
        """Record `path` read for must-read-first; failure is benign (fails closed in status)."""
        try:
            resolved = path.resolve()
        except OSError:
            return
        try:
            st = resolved.stat()
        except OSError:
            return
        self._read_records[resolved] = ReadRecord(
            mtime=st.st_mtime, size=st.st_size, partial=partial,
        )

    def read_status(
        self, path: Path,
    ) -> Literal["never_read", "stale", "partial", "fresh"]:
        """Read-state vs session: stale = fingerprint differs or path gone; partial = sliced."""
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
        """Walk from resolved_path.parent up to (but not including) self._cwd."""
        # touched path at or above cwd → parent escapes cwd → no work.
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

    def build(self, history: list[BaseMessage]) -> list[BaseMessage]:
        # Anthropic-only cache_control breakpoints: system / project-memory / skills.
        is_anthropic = _is_anthropic_provider(self._model)
        breakpoint_count = 0

        sys_msg = SystemMessage(self._system_prompt)
        if is_anthropic:
            sys_msg.additional_kwargs["cache_control"] = {"type": "ephemeral"}
            breakpoint_count += 1
        messages: list[BaseMessage] = [sys_msg]

        eager = _joined_eager(self._primary_memory, self._rules.unconditional)
        if eager:
            project_memory_msg = SystemMessage(
                "<system-reminder>\n"
                f"{_OVERRIDE_PREAMBLE}\n\n"
                f"<project-memory>\n{eager}\n</project-memory>\n"
                "</system-reminder>"
            )
            if is_anthropic:
                project_memory_msg.additional_kwargs["cache_control"] = {
                    "type": "ephemeral",
                }
                breakpoint_count += 1
            messages.append(project_memory_msg)

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

        # Sorted by name so set-preserving registry reorderings don't churn the
        # prompt-cache prefix. Hides user-only and unactivated-conditional skills.
        visible_skills = sorted(
            (
                s for s in self._skills_available
                if not s.disable_model_invocation and not s.is_conditional()
            ),
            key=lambda s: s.name,
        )
        if visible_skills:
            available_lines: list[str] = []
            for s in visible_skills:
                line = f"- {s.name}: {s.description}"
                if s.when_to_use:
                    line += f" [when to use: {s.when_to_use}]"
                available_lines.append(line)
            skills_msg = HumanMessage(
                "<skills-available>\n"
                + "\n".join(available_lines)
                + "\n</skills-available>"
            )
            if is_anthropic:
                skills_msg.additional_kwargs["cache_control"] = {
                    "type": "ephemeral",
                }
                breakpoint_count += 1
            messages.append(skills_msg)
        for skill in self._invoked_skills:
            messages.append(
                HumanMessage(
                    f'<skill-invoked name="{skill.name}">\n'
                    f"{skill.body}\n"
                    "</skill-invoked>"
                )
            )

        if self._todos_provider is not None:
            todos = self._todos_provider()
            if todos:
                body = _render_todos_body(todos)
                messages.append(HumanMessage(f"<todos>\n{body}\n</todos>"))

        # Drain on every build so the prompt envelope flushes the queue.
        if self._notifications_drainer is not None:
            drained = list(self._notifications_drainer())
            if drained:
                cap = 5
                head = drained[-cap:] if len(drained) > cap else drained
                lines: list[str] = []
                for n in head:
                    line = f"- {n.task_id[:8]} [{n.status}] {n.description}"
                    if n.summary:
                        line += f": {n.summary}"
                    lines.append(line)
                if len(drained) > cap:
                    lines.append(f"({len(drained) - cap} more earlier)")
                body = "\n".join(lines)
                messages.append(HumanMessage(
                    f"<task-notification>\n{body}\n</task-notification>"
                ))

        messages.extend(history)

        if is_anthropic and breakpoint_count > 0:
            journal.write(
                "cache_breakpoints_set",
                count=breakpoint_count,
                provider=_provider_type(self._model),
            )
        return messages


def _joined_eager(primary: str, unconditional: list[Rule]) -> str:
    pieces: list[str] = []
    if primary:
        pieces.append(primary)
    for rule in unconditional:
        if rule.content:
            pieces.append(rule.content)
    return "\n\n".join(pieces)


def _provider_type(model: BaseChatModel | None) -> str:
    if model is None:
        return ""
    try:
        return str(model._llm_type)
    except Exception:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
        return ""


def _is_anthropic_provider(model: BaseChatModel | None) -> bool:
    # Prefix match accepts future Anthropic-shaped wrappers (e.g. routing proxies).
    return _provider_type(model).lower().startswith("anthropic")


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _render_todos_body(todos: list[TodoItem]) -> str:
    lines: list[str] = []
    for t in todos:
        if t.status == "completed":
            lines.append(f"- [completed] {t.content}")
        else:
            lines.append(f"- [{t.status}] {t.content} (active: {t.active_form})")
    return "\n".join(lines)


def _intermediate_dirs(start: Path, cwd: Path) -> list[Path]:
    """Outer→inner dirs from `start` up to (excluding) `cwd`; caller ensured `start < cwd`."""
    chain: list[Path] = []
    current = start
    while current != cwd:
        chain.append(current)
        current = current.parent
    chain.reverse()
    return chain
