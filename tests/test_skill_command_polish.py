"""Tests for skill-command UX polish (v0.12).

Five polish items covered here (one section per item):

1. ``user_invocable=False`` filter wired to slash-command registration.
2. Slash-path errors on missing required arguments (match tool path).
3. ``/help`` grouping by source.
4. ``/help`` multi-line description collapse.
5. ``skill_invoked`` journal event carries ``allowed_tools`` + ``source``
   (Skill.layer) at invocation time (honest audit trail while runtime
   enforcement remains deferred to v0.13).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aura.cli.commands import build_default_registry
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.commands import CommandRegistry
from aura.core.commands.builtin import HelpCommand
from aura.core.commands.registry import CommandRegistry as _Reg
from aura.core.commands.types import CommandResult, CommandSource
from aura.core.persistence import journal as journal_module
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.command import SkillCommand
from aura.core.skills.types import Skill
from aura.tools.skill import SkillTool
from tests.conftest import FakeChatModel

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _skill(
    name: str = "foo",
    *,
    description: str | None = None,
    body: str = "body",
    layer: str = "user",
    arguments: tuple[str, ...] = (),
    user_invocable: bool = True,
    disable_model_invocation: bool = False,
    allowed_tools: frozenset[str] = frozenset(),
) -> Skill:
    return Skill(
        name=name,
        description=description or f"Description of {name}.",
        body=body,
        source_path=Path(f"/tmp/{name}.md"),
        layer=layer,  # type: ignore[arg-type]
        arguments=arguments,
        user_invocable=user_invocable,
        disable_model_invocation=disable_model_invocation,
        allowed_tools=allowed_tools,
    )


def _agent(tmp_path: Path, skills: list[Skill] | None = None) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "db"),
    )
    if skills:
        # Inject into the agent's own registry so build_default_registry
        # finds them on the live agent (same path the production CLI uses).
        for s in skills:
            agent._skill_registry.register(s)
    return agent


@pytest.fixture
def journal_path(tmp_path: Path) -> Iterator[Path]:
    """Configure the journal to write to a tmp file and reset after."""
    log = tmp_path / "events.jsonl"
    journal_module.configure(log)
    yield log
    journal_module.reset()


def _journal_events(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ===========================================================================
# Item 1 — user_invocable=False filter
# ===========================================================================


def test_user_invocable_false_skill_not_registered_as_slash_command(
    tmp_path: Path,
) -> None:
    """Skill with ``user_invocable: false`` must NOT get a slash command.

    Claude-code hides these from the slash-command picker while keeping
    them model-invocable. Before this fix we iterated ``registry.list()``
    which returned ALL skills; the fix swaps to the filtered accessor.
    """
    hidden = _skill("hidden-from-slash", user_invocable=False)
    visible = _skill("visible", user_invocable=True)
    agent = _agent(tmp_path, [hidden, visible])
    try:
        r = build_default_registry(agent=agent)
        names = {c.name for c in r.list()}
        assert "/visible" in names
        assert "/hidden-from-slash" not in names
    finally:
        agent.close()


def test_user_invocable_false_skill_still_model_invocable(
    tmp_path: Path,
) -> None:
    """Same skill must still be reachable through the model-facing SkillTool."""
    hidden = _skill("model-only", user_invocable=False)
    agent = _agent(tmp_path, [hidden])
    try:
        captured: list[Skill] = []
        tool = SkillTool(
            recorder=captured.append,
            registry=agent._skill_registry,
            session_id_provider=lambda: "sid",
        )
        result = tool._run(name="model-only")
        assert result["invoked"] is True
        assert captured and captured[0].name == "model-only"
    finally:
        agent.close()


def test_disable_model_invocation_true_skill_hidden_from_tool(
    tmp_path: Path,
) -> None:
    """``disable_model_invocation: true`` blocks the model-facing tool path."""
    model_blocked = _skill("user-only", disable_model_invocation=True)
    agent = _agent(tmp_path, [model_blocked])
    try:
        tool = SkillTool(
            recorder=lambda _s: None,
            registry=agent._skill_registry,
            session_id_provider=lambda: "sid",
        )
        from aura.schemas.tool import ToolError

        with pytest.raises(ToolError):
            tool._run(name="user-only")
    finally:
        agent.close()


def test_fully_hidden_skill(tmp_path: Path) -> None:
    """user_invocable=False + disable_model_invocation=True → nowhere."""
    ghost = _skill(
        "ghost", user_invocable=False, disable_model_invocation=True,
    )
    agent = _agent(tmp_path, [ghost])
    try:
        r = build_default_registry(agent=agent)
        names = {c.name for c in r.list()}
        assert "/ghost" not in names

        tool = SkillTool(
            recorder=lambda _s: None,
            registry=agent._skill_registry,
            session_id_provider=lambda: "sid",
        )
        from aura.schemas.tool import ToolError

        with pytest.raises(ToolError):
            tool._run(name="ghost")
    finally:
        agent.close()


# ===========================================================================
# Item 2 — slash path errors on missing required args
# ===========================================================================


@pytest.mark.asyncio
async def test_slash_path_errors_on_missing_required_arg(
    tmp_path: Path,
) -> None:
    """Declared args + empty invocation → printed error, not silent success."""
    skill = _skill("needsarg", arguments=("topic",))
    agent = _agent(tmp_path)
    try:
        cmd = SkillCommand(skill=skill, agent=agent)
        result = await cmd.handle("", agent)
        assert result.handled is True
        assert result.kind == "print"
        assert "missing" in result.text
        assert "topic" in result.text
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_slash_path_accepts_exact_arg_count(tmp_path: Path) -> None:
    """Exact-count invocation still records the skill invocation."""
    skill = _skill("needsarg", arguments=("topic",))
    agent = _agent(tmp_path)
    try:
        cmd = SkillCommand(skill=skill, agent=agent)
        result = await cmd.handle("foo", agent)
        assert result.handled is True
        assert result.kind == "print"
        assert "missing" not in result.text
        messages = agent._context.build([])
        contents = " ".join(str(m.content) for m in messages)
        assert '<skill-invoked name="needsarg">' in contents
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_slash_path_accepts_more_args_than_declared(
    tmp_path: Path,
) -> None:
    """Extra positionals beyond declared arity are ignored (claude-code parity)."""
    skill = _skill("onearg", arguments=("topic",))
    agent = _agent(tmp_path)
    try:
        cmd = SkillCommand(skill=skill, agent=agent)
        result = await cmd.handle("foo bar baz", agent)
        assert result.handled is True
        assert result.kind == "print"
        assert "missing" not in result.text
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_tool_and_slash_paths_share_error_format(
    tmp_path: Path,
) -> None:
    """Tool + slash paths produce identical missing-arg error text."""
    skill = _skill("shared", arguments=("alpha", "beta"))
    agent = _agent(tmp_path, [skill])
    try:
        # Slash path: handle("") → error text in result.text
        cmd = SkillCommand(skill=skill, agent=agent)
        slash_result = await cmd.handle("", agent)
        assert slash_result.kind == "print"

        # Tool path: raises ToolError with the same message.
        from aura.schemas.tool import ToolError

        tool = SkillTool(
            recorder=lambda _s: None,
            registry=agent._skill_registry,
            session_id_provider=lambda: "sid",
        )
        with pytest.raises(ToolError) as exc_info:
            tool._run(name="shared", arguments=[])
        assert str(exc_info.value) == slash_result.text
    finally:
        await agent.aclose()


# ===========================================================================
# Item 3 — /help grouping by source
# ===========================================================================


class _StubCommand:
    def __init__(
        self,
        name: str,
        *,
        source: CommandSource = "builtin",
        description: str = "stub",
        allowed_tools: tuple[str, ...] = (),
        argument_hint: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.source: CommandSource = source
        self.allowed_tools: tuple[str, ...] = allowed_tools
        self.argument_hint: str | None = argument_hint

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        return CommandResult(handled=True, kind="print", text="ok")


@pytest.mark.asyncio
async def test_help_groups_by_source(tmp_path: Path) -> None:
    """``/help`` groups by source: Builtins → Skills → MCP."""
    r: _Reg = CommandRegistry()
    help_cmd = HelpCommand(registry=r)
    r.register(help_cmd)
    r.register(_StubCommand("/builtin-a", source="builtin"))
    r.register(_StubCommand("/skill-a", source="skill"))
    r.register(_StubCommand("/mcp-a", source="mcp"))
    agent = _agent(tmp_path)
    try:
        result = await r.dispatch("/help", agent)
        text = result.text
        # Section headers must appear in order: Builtins → Skills → MCP.
        b_idx = text.index("Builtins:")
        s_idx = text.index("Skills:")
        m_idx = text.index("MCP:")
        assert b_idx < s_idx < m_idx

        # Commands land in the correct section (use indices).
        assert text.index("/builtin-a") < s_idx
        assert s_idx < text.index("/skill-a") < m_idx
        assert m_idx < text.index("/mcp-a")
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_help_skips_empty_group(tmp_path: Path) -> None:
    """Empty groups are omitted from the output."""
    r: _Reg = CommandRegistry()
    help_cmd = HelpCommand(registry=r)
    r.register(help_cmd)
    r.register(_StubCommand("/only-builtin", source="builtin"))
    agent = _agent(tmp_path)
    try:
        result = await r.dispatch("/help", agent)
        assert "Builtins:" in result.text
        assert "Skills:" not in result.text
        assert "MCP:" not in result.text
    finally:
        agent.close()


@pytest.mark.asyncio
async def test_help_maintains_alignment_within_group(tmp_path: Path) -> None:
    """The 24-col label column is preserved within each group."""
    r: _Reg = CommandRegistry()
    help_cmd = HelpCommand(registry=r)
    r.register(help_cmd)
    r.register(_StubCommand("/a", source="skill", description="short desc"))
    r.register(
        _StubCommand(
            "/loooong-name",
            source="skill",
            description="long desc",
            argument_hint="<x>",
        )
    )
    agent = _agent(tmp_path)
    try:
        result = await r.dispatch("/help", agent)
        lines = result.text.splitlines()
        # Locate the Skills: section and take command lines until the next
        # blank line / section header.
        skill_idx = next(
            i for i, ln in enumerate(lines) if ln.strip() == "Skills:"
        )
        skill_lines: list[str] = []
        for ln in lines[skill_idx + 1:]:
            if not ln.strip() or ln.strip().endswith(":"):
                break
            skill_lines.append(ln)
        assert len(skill_lines) == 2
        # Every skill line pads the label column to a consistent width;
        # the description must therefore start at the same column across
        # the two lines.
        desc_cols = []
        for ln in skill_lines:
            for token in ("short desc", "long desc"):
                idx = ln.find(token)
                if idx != -1:
                    desc_cols.append(idx)
                    break
        assert len(desc_cols) == 2
        assert desc_cols[0] == desc_cols[1], (
            f"description column drift in group: {desc_cols} in {skill_lines}"
        )
    finally:
        agent.close()


# ===========================================================================
# Item 4 — /help multi-line description collapse
# ===========================================================================


@pytest.mark.asyncio
async def test_help_collapses_multiline_description(tmp_path: Path) -> None:
    """Multi-line description must collapse to its first non-empty line."""
    r: _Reg = CommandRegistry()
    help_cmd = HelpCommand(registry=r)
    r.register(help_cmd)
    r.register(
        _StubCommand(
            "/multi",
            source="skill",
            description="first line only\n\nsecond paragraph should be gone",
        )
    )
    agent = _agent(tmp_path)
    try:
        result = await r.dispatch("/help", agent)
        # The multi line within the description must not leak into /help.
        assert "second paragraph should be gone" not in result.text
        assert "first line only" in result.text

        # Find the /multi line and assert no embedded newline between the
        # name and the description (a naive f-string would leak \n\n).
        lines = result.text.splitlines()
        multi_lines = [ln for ln in lines if "/multi" in ln]
        assert len(multi_lines) == 1
        assert "first line only" in multi_lines[0]
    finally:
        agent.close()


# ===========================================================================
# Item 5 — skill_invoked journal event carries allowed_tools + source
# ===========================================================================


@pytest.mark.asyncio
async def test_skill_invoked_journal_event_records_allowed_tools(
    tmp_path: Path, journal_path: Path,
) -> None:
    """Slash-invoked skill → journal event with sorted ``allowed_tools``."""
    skill = _skill(
        "audited",
        allowed_tools=frozenset({"read_file", "grep"}),
    )
    agent = _agent(tmp_path)
    try:
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)
        events = _journal_events(journal_path)
        invoked = [e for e in events if e.get("event") == "skill_invoked"]
        assert len(invoked) == 1
        ev = invoked[0]
        assert ev["name"] == "audited"
        # Stable ordering: sorted tuple of names.
        assert ev["allowed_tools"] == ["grep", "read_file"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_skill_invoked_journal_event_includes_source(
    tmp_path: Path, journal_path: Path,
) -> None:
    """Event carries ``source`` sourced from ``Skill.layer`` (user|project|managed)."""
    skill = _skill("layered", layer="project")
    agent = _agent(tmp_path)
    try:
        cmd = SkillCommand(skill=skill, agent=agent)
        await cmd.handle("", agent)
        events = _journal_events(journal_path)
        invoked = [e for e in events if e.get("event") == "skill_invoked"]
        assert len(invoked) == 1
        assert invoked[0]["source"] == "project"
    finally:
        await agent.aclose()
