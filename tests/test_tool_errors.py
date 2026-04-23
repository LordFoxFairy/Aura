"""Tests for hint-propagation to the LLM's ToolMessage + bash progress events.

Feature 1: ``aura.cli.render._hint_for_error`` used to be a UI-only affair —
the hint went to the red Panel but the model saw only the raw error. These
tests lock in the fix: on a failing tool, the SAME hint text surfaces in
the ToolMessage content the model reads next turn.

Feature 2: ``aura.tools.bash`` now streams stdout/stderr chunks via the
``ToolCallProgress`` event while the subprocess is still running, so a
long ``npm test`` no longer sits silently for 30s+ before the first line
appears.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.registry import ToolRegistry
from aura.schemas.events import (
    AgentEvent,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.schemas.tool import ToolError
from aura.tools.base import build_tool
from aura.tools.errors import hint_for_error
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


# ---------------------------------------------------------------------------
# Feature 1 — hint propagation
# ---------------------------------------------------------------------------
class _P(BaseModel):
    msg: str = ""


def _make_failing_tool(error_text: str) -> Any:
    def _run(msg: str = "") -> dict[str, Any]:
        raise ToolError(error_text)

    return build_tool(
        name="faker",
        description="always fails",
        args_schema=_P,
        func=_run,
        is_read_only=True,
        is_concurrency_safe=True,
    )


async def _run_one_turn_with_failure(tool: Any) -> list[BaseMessage]:
    tool_calls = [{"name": "faker", "args": {}, "id": "tc_1"}]
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry([tool])
    loop = AgentLoop(
        model=model,
        registry=registry,
        context=make_minimal_context(),
        hooks=HookChain(),
    )
    history: list[BaseMessage] = []
    async for _ in loop.run_turn(user_prompt="go", history=history):
        pass
    return history


def _tool_message_content(history: list[BaseMessage]) -> str:
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    content = tool_msgs[0].content
    assert isinstance(content, str)
    return content


@pytest.mark.asyncio
async def test_hint_for_ripgrep_missing_appears_in_tool_message() -> None:
    tool = _make_failing_tool("ripgrep (rg) is not installed on PATH")
    history = await _run_one_turn_with_failure(tool)
    content = _tool_message_content(history)
    expected_hint = hint_for_error("faker", "ripgrep is not installed")
    assert expected_hint is not None
    assert expected_hint in content
    # The original error text must still be present — the hint is additive,
    # not a replacement.
    assert "ripgrep" in content


@pytest.mark.asyncio
async def test_hint_for_must_read_first_appears_in_tool_message() -> None:
    tool = _make_failing_tool("file has not been read yet: /tmp/foo.py")
    history = await _run_one_turn_with_failure(tool)
    content = _tool_message_content(history)
    assert "Hint:" in content
    assert "read_file" in content


@pytest.mark.asyncio
async def test_hint_for_not_found_appears_in_tool_message() -> None:
    tool = _make_failing_tool("not found: /tmp/missing.py")
    history = await _run_one_turn_with_failure(tool)
    content = _tool_message_content(history)
    assert "Hint:" in content
    # The "not found" hint talks about the path being moved/deleted.
    assert "path" in content.lower()


@pytest.mark.asyncio
async def test_no_hint_means_error_text_unchanged() -> None:
    tool = _make_failing_tool("surprise entirely novel failure xyz123")
    history = await _run_one_turn_with_failure(tool)
    content = _tool_message_content(history)
    assert "Hint:" not in content
    assert "surprise entirely novel failure" in content


# ---------------------------------------------------------------------------
# Feature 2 — bash stdout/stderr streaming
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bash_emits_progress_events_for_stdout() -> None:
    # A bash call that prints multiple lines should surface as
    # ToolCallProgress events in the loop's event stream BEFORE the
    # ToolCallCompleted. Verifies the contextvar plumbing end-to-end.
    from aura.tools.bash import bash

    tool_calls = [
        {
            "name": "bash",
            "args": {"command": "echo line-one; echo line-two"},
            "id": "tc_1",
        },
    ]
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry([bash])
    loop = AgentLoop(
        model=model,
        registry=registry,
        context=make_minimal_context(),
        hooks=HookChain(),
    )

    events: list[AgentEvent] = []
    history: list[BaseMessage] = []
    async for ev in loop.run_turn(user_prompt="go", history=history):
        events.append(ev)

    progress = [e for e in events if isinstance(e, ToolCallProgress)]
    completed = [e for e in events if isinstance(e, ToolCallCompleted)]

    assert len(completed) == 1
    assert len(progress) >= 1, "expected at least one ToolCallProgress from bash"
    assert all(p.name == "bash" for p in progress)
    combined = "".join(p.chunk for p in progress if p.stream == "stdout")
    assert "line-one" in combined
    assert "line-two" in combined

    # Ordering: every ToolCallStarted precedes the progress stream, which
    # precedes the Completed — the loop must not buffer progress after the
    # tool returns.
    ix_started = next(
        i for i, e in enumerate(events) if isinstance(e, ToolCallStarted)
    )
    ix_progress = next(
        i for i, e in enumerate(events) if isinstance(e, ToolCallProgress)
    )
    ix_completed = next(
        i for i, e in enumerate(events) if isinstance(e, ToolCallCompleted)
    )
    assert ix_started < ix_progress < ix_completed


@pytest.mark.asyncio
async def test_bash_emits_progress_events_for_stderr() -> None:
    # stderr path mirrors stdout — ensure the ``stream`` label flows
    # through so the renderer can style the two independently.
    from aura.tools.bash import bash

    tool_calls = [
        {
            "name": "bash",
            "args": {"command": "echo oopsie 1>&2"},
            "id": "tc_1",
        },
    ]
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=tool_calls)),
        FakeTurn(message=AIMessage(content="done")),
    ])
    registry = ToolRegistry([bash])
    loop = AgentLoop(
        model=model,
        registry=registry,
        context=make_minimal_context(),
        hooks=HookChain(),
    )

    events: list[AgentEvent] = []
    history: list[BaseMessage] = []
    async for ev in loop.run_turn(user_prompt="go", history=history):
        events.append(ev)

    stderr_chunks = [
        e for e in events
        if isinstance(e, ToolCallProgress) and e.stream == "stderr"
    ]
    assert stderr_chunks, "expected at least one stderr ToolCallProgress"
    assert "oopsie" in "".join(p.chunk for p in stderr_chunks)


def test_renderer_prints_progress_chunk_dim() -> None:
    import io

    from rich.console import Console

    from aura.cli.render import Renderer

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, highlight=False)
    renderer = Renderer(console)
    renderer.on_event(ToolCallProgress(
        name="bash", stream="stdout", chunk="hello from the subprocess\n",
    ))
    out = buf.getvalue()
    assert "hello from the subprocess" in out
    # The renderer prefixes each line with ``│`` so progress visually nests
    # under the started marker rather than masquerading as assistant text.
    assert "│" in out
