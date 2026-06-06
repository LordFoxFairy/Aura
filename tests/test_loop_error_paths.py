"""Tests for AgentLoop defensive error paths in tool dispatch."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from aura.application.compact import MicrocompactPolicy
from aura.application.hooks import HookChain
from aura.application.loop import AgentLoop
from aura.domain.abort import AbortController, AbortException
from aura.domain.events import AgentEvent, Final, ToolCallCompleted
from aura.domain.permission.outcome import Allow, Ask, Outcome
from aura.domain.tool import ToolMetadata, ToolResult
from aura.domain.tool_registry import ToolRegistry
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn, make_minimal_context


class _EchoParams(BaseModel):
    msg: str


def _echo(msg: str) -> dict[str, Any]:
    return {"echoed": msg}


_echo_tool: BaseTool = build_tool(
    name="echo",
    description="echoes input",
    args_schema=_EchoParams,
    func=_echo,
    is_read_only=True,
    is_concurrency_safe=True,
)


def _explode(msg: str) -> dict[str, Any]:
    raise RuntimeError("kaboom")


_exploding_tool: BaseTool = build_tool(
    name="exploder",
    description="always raises",
    args_schema=_EchoParams,
    func=_explode,
    is_destructive=True,
)


@pytest.mark.asyncio
async def test_unknown_tool_name_emits_error_tool_message_and_continues() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "ghost", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="recovered")),
    ])
    registry = ToolRegistry([_echo_tool])
    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )

    history: list[BaseMessage] = []
    events: list[AgentEvent] = []
    history.append(HumanMessage(content="go"))
    async for ev in loop.run_turn(history=history):
        events.append(ev)

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc_1"
    assert tool_msgs[0].status == "error"
    assert "unknown tool" in str(tool_msgs[0].content)
    assert "ghost" in str(tool_msgs[0].content)

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].error is not None

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert finals[0].message == "recovered"


@pytest.mark.asyncio
async def test_invalid_args_emit_error_tool_message_and_continues() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"wrong_field": 1}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="recovered")),
    ])
    registry = ToolRegistry([_echo_tool])
    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )

    history: list[BaseMessage] = []
    events: list[AgentEvent] = []
    history.append(HumanMessage(content="go"))
    async for ev in loop.run_turn(history=history):
        events.append(ev)

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "tc_1"
    assert tool_msgs[0].status == "error"
    assert "invalid args" in str(tool_msgs[0].content)


@pytest.mark.asyncio
async def test_invoke_exception_emits_error_tool_message_with_exception_info() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "exploder", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="noted")),
    ])
    registry = ToolRegistry([_exploding_tool])
    loop = AgentLoop(
        model=model, registry=registry, context=make_minimal_context(),
        hooks=HookChain(),
    )

    history: list[BaseMessage] = []
    events: list[AgentEvent] = []
    history.append(HumanMessage(content="go"))
    async for ev in loop.run_turn(history=history):
        events.append(ev)

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    content = str(tool_msgs[0].content)
    assert "RuntimeError" in content
    assert "kaboom" in content


@pytest.mark.asyncio
async def test_post_tool_sees_exception_result() -> None:
    seen: list[ToolResult] = []

    async def capture(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: object,
        **_: object,
    ) -> ToolResult:
        seen.append(result)
        return result

    hooks = HookChain(post_tool=[capture])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "exploder", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="noted")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_exploding_tool]),
        context=make_minimal_context(), hooks=hooks,
    )

    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].error is not None
    assert "RuntimeError" in seen[0].error


@pytest.mark.asyncio
async def test_pre_tool_not_fired_for_unknown_tool() -> None:
    calls: list[str] = []

    async def record(
        *, tool: BaseTool, args: dict[str, Any], state: object, **_: object
    ) -> Outcome:
        from aura.domain.permission.decision import Decision
        calls.append(tool.name)
        return Allow(decision=Decision(allow=True, reason="mode_bypass"))

    hooks = HookChain(pre_tool=[record])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "ghost", "args": {"msg": "x"}, "id": "tc_1"}
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]),
        context=make_minimal_context(), hooks=hooks,
    )

    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert calls == []


@pytest.mark.asyncio
async def test_mixed_tool_calls_each_produce_own_tool_message() -> None:
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"msg": "good"}, "id": "tc_1"},
            {"name": "ghost", "args": {"msg": "x"}, "id": "tc_2"},
            {"name": "echo", "args": {"wrong_field": 1}, "id": "tc_3"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    history: list[BaseMessage] = []
    history.append(HumanMessage(content="go"))
    async for _ in loop.run_turn(history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 3
    assert [m.tool_call_id for m in tool_msgs] == ["tc_1", "tc_2", "tc_3"]
    assert tool_msgs[0].status == "success"
    assert tool_msgs[1].status == "error"
    assert tool_msgs[2].status == "error"
    assert "unknown tool" in str(tool_msgs[1].content)
    assert "invalid args" in str(tool_msgs[2].content)


# ---------------------------------------------------------------------------
# Helpers for the residual-branch suite below.
# ---------------------------------------------------------------------------


def _length_turn(content: str, reason: str) -> FakeTurn:
    """A scripted AIMessage carrying a provider truncation finish_reason."""
    return FakeTurn(
        message=AIMessage(content=content, response_metadata={"finish_reason": reason}),
    )


def _raw_schema_tool(seen: dict[str, object]) -> BaseTool:
    """Tool whose ``args_schema`` is a JSON dict, not a Pydantic model.

    Reaches the ``_validated_args_model -> None`` branch: the loop must skip
    schema validation entirely and invoke with the raw, unvalidated args.
    """
    def _capture(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"got": dict(kwargs)}

    tool: BaseTool = StructuredTool.from_function(
        func=_capture,
        name="raw",
        description="dict-schema tool",
        args_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "title": "Raw",
        },
    )
    meta = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=False,
        rule_matcher=None,
        args_preview=None,
        timeout_sec=None,
        max_result_size_chars=None,
    )
    object.__setattr__(tool, "aura_metadata", meta)
    return tool


class _OverflowThenOkModel(FakeChatModel):
    """Raises a context-overflow error on the first ainvoke, then succeeds.

    Drives the reactive-compaction recovery loop: the first ``_invoke_model``
    attempt overflows, the compact callback shrinks history, the retry passes.
    """

    def __init__(self, ok_message: AIMessage, **kwargs: Any) -> None:
        super().__init__(turns=[], **kwargs)
        self.__dict__["_ok_message"] = ok_message
        self.__dict__["_raised"] = False

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        if not self.__dict__["_raised"]:
            self.__dict__["_raised"] = True
            raise ValueError("prompt is too long for the model context window")
        ok: AIMessage = self.__dict__["_ok_message"]
        return ChatResult(generations=[ChatGeneration(message=ok)])


class _CapturingModel(FakeChatModel):
    """FakeChatModel that records each outgoing message view before replying."""

    def __init__(self, turns: list[FakeTurn], sink: list[list[BaseMessage]]) -> None:
        super().__init__(turns=turns)
        self.__dict__["_sink"] = sink

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_: Any,
    ) -> ChatResult:
        sink: list[list[BaseMessage]] = self.__dict__["_sink"]
        sink.append(list(messages))
        return await super()._agenerate(messages, stop, None)


def _blocking_tool(gate: asyncio.Event) -> BaseTool:
    """Concurrency-safe tool that never returns until ``gate`` is set.

    Lets a per-batch deadline elapse so the batch-timeout synthesiser fires.
    """

    async def _hang() -> dict[str, object]:
        await gate.wait()
        return {"ok": True}

    class _Empty(BaseModel):
        pass

    return build_tool(
        name="hang",
        description="blocks until gate set",
        args_schema=_Empty,
        coroutine=_hang,
        is_read_only=True,
        is_concurrency_safe=True,
    )


# ---------------------------------------------------------------------------
# Residual branch coverage.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_dict_schema_tool_skips_validation_and_runs() -> None:
    """A JSON-dict args_schema must not crash planning; raw args flow through."""
    seen: dict[str, object] = {}
    tool = _raw_schema_tool(seen)
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "raw", "args": {"msg": "hi", "extra": 9}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    async for _ in loop.run_turn(history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "success"
    # Unvalidated: the extra field that a Pydantic forbid-extra schema would
    # have rejected is forwarded verbatim into the tool invocation.
    assert seen == {"msg": "hi", "extra": 9}


@pytest.mark.asyncio
async def test_ask_outcome_leak_denies_with_unresolved_message() -> None:
    """An Ask that escapes the hook chain is a fail-closed deny, not a run."""
    ran: list[str] = []

    def _echo(msg: str) -> dict[str, Any]:
        ran.append(msg)
        return {"echoed": msg}

    echo_tool: BaseTool = build_tool(
        name="echo",
        description="echoes",
        args_schema=_EchoParams,
        func=_echo,
        is_read_only=True,
    )

    async def _leak_ask(
        *, tool: BaseTool, args: dict[str, Any], state: object, **_: object,
    ) -> Outcome:
        return Ask(reason="needs confirmation")

    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"msg": "x"}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([echo_tool]),
        context=make_minimal_context(), hooks=HookChain(pre_tool=[_leak_ask]),
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    async for _ in loop.run_turn(history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    assert "permission escalation unresolved" in str(tool_msgs[0].content)
    assert ran == []


@pytest.mark.asyncio
async def test_reactive_compact_callback_recovers_from_overflow() -> None:
    """Context overflow must trigger the compact callback then retry once."""
    compacted: list[int] = []

    async def _compact(history: list[BaseMessage]) -> None:
        compacted.append(len(history))

    model = _OverflowThenOkModel(ok_message=AIMessage(content="recovered"))
    loop = AgentLoop(
        model=model, registry=ToolRegistry([]),
        context=make_minimal_context(), hooks=HookChain(),
        compact_callback=_compact,
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    finals: list[Final] = []
    async for ev in loop.run_turn(history=history):
        if isinstance(ev, Final):
            finals.append(ev)

    assert len(compacted) == 1
    assert model.ainvoke_calls == 2
    assert len(finals) == 1
    assert finals[0].message == "recovered"


@pytest.mark.asyncio
async def test_reactive_compact_only_retries_once_then_propagates() -> None:
    """A second consecutive overflow exceeds the one-shot budget and raises."""
    calls: list[int] = []

    class _AlwaysOverflow(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "always-overflow"

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: object | None = None,
            **_: Any,
        ) -> ChatResult:
            calls.append(1)
            raise ValueError("prompt is too long")

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: object | None = None,
            **_: Any,
        ) -> ChatResult:
            raise NotImplementedError

    async def _compact(history: list[BaseMessage]) -> None:
        return None

    loop = AgentLoop(
        model=_AlwaysOverflow(), registry=ToolRegistry([]),
        context=make_minimal_context(), hooks=HookChain(),
        compact_callback=_compact,
    )

    with pytest.raises(ValueError, match="prompt is too long"):
        async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
            pass

    # Initial attempt + exactly one post-compaction retry, then give up.
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_microcompact_policy_clears_old_pairs_in_view() -> None:
    """Policy-driven microcompact must blank stale tool payloads view-side."""
    history: list[BaseMessage] = [HumanMessage(content="start")]
    for i in range(8):
        history.append(AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": f"/f{i}"}, "id": f"tc-{i}"},
        ]))
        history.append(ToolMessage(
            content=f"payload-{i}", tool_call_id=f"tc-{i}",
            name="read_file", status="success",
        ))

    captured: list[list[BaseMessage]] = []
    model = _CapturingModel(
        turns=[FakeTurn(message=AIMessage(content="ok"))], sink=captured,
    )
    policy = MicrocompactPolicy(trigger_pairs=5, keep_recent=3)
    loop = AgentLoop(
        model=model, registry=ToolRegistry([]),
        context=make_minimal_context(), hooks=HookChain(),
        microcompact_policy=policy,
    )

    async for _ in loop.run_turn(history=history):
        pass

    sent = captured[0]
    cleared = [
        m for m in sent
        if isinstance(m, ToolMessage) and m.content == "[Old tool result content cleared]"
    ]
    # 8 pairs, keep 3 recent → the 5 oldest tool payloads are blanked.
    assert len(cleared) == 5
    # Stored history stays full — microcompact is view-only.
    stored = [m for m in history if isinstance(m, ToolMessage)]
    assert all(str(m.content).startswith("payload-") for m in stored)


@pytest.mark.asyncio
async def test_microcompact_policy_below_trigger_keeps_full_view() -> None:
    """Below the trigger threshold the view passes through unmodified."""
    history: list[BaseMessage] = [HumanMessage(content="start")]
    for i in range(2):
        history.append(AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": f"/f{i}"}, "id": f"tc-{i}"},
        ]))
        history.append(ToolMessage(
            content=f"payload-{i}", tool_call_id=f"tc-{i}",
            name="read_file", status="success",
        ))

    captured: list[list[BaseMessage]] = []
    model = _CapturingModel(
        turns=[FakeTurn(message=AIMessage(content="ok"))], sink=captured,
    )
    loop = AgentLoop(
        model=model, registry=ToolRegistry([]),
        context=make_minimal_context(), hooks=HookChain(),
        microcompact_policy=MicrocompactPolicy(trigger_pairs=5, keep_recent=3),
    )

    async for _ in loop.run_turn(history=history):
        pass

    cleared = [
        m for m in captured[0]
        if isinstance(m, ToolMessage) and m.content == "[Old tool result content cleared]"
    ]
    assert cleared == []


@pytest.mark.asyncio
async def test_batch_timeout_synthesises_error_result() -> None:
    """A tool exceeding the per-batch deadline yields a synthetic timeout error."""
    gate = asyncio.Event()
    tool = _blocking_tool(gate)
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "hang", "args": {}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="moved on")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([tool]),
        context=make_minimal_context(), hooks=HookChain(),
        batch_timeout_sec=0.05,
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    try:
        async for _ in loop.run_turn(history=history):
            pass
    finally:
        gate.set()

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "error"
    assert "batch timeout after 0.05s" in str(tool_msgs[0].content)


@pytest.mark.asyncio
async def test_batch_timeout_runs_post_tool_on_synthesised_result() -> None:
    """The timeout cancel path still feeds post_tool a uniform ToolResult shape."""
    gate = asyncio.Event()
    tool = _blocking_tool(gate)
    seen: list[ToolResult] = []

    async def _capture(
        *, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: object,
        **_: object,
    ) -> ToolResult:
        seen.append(result)
        return result

    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "hang", "args": {}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="moved on")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([tool]),
        context=make_minimal_context(), hooks=HookChain(post_tool=[_capture]),
        batch_timeout_sec=0.05,
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    try:
        async for _ in loop.run_turn(history=history):
            pass
    finally:
        gate.set()

    assert len(seen) == 1
    assert seen[0].ok is False
    assert seen[0].error is not None
    assert "batch timeout" in seen[0].error


@pytest.mark.asyncio
async def test_malformed_batch_timeout_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-numeric AURA_BATCH_TIMEOUT_SEC must not crash the hot-path ctor."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "not-a-number")
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"msg": "ok"}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    async for _ in loop.run_turn(history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "success"


@pytest.mark.asyncio
async def test_zero_batch_timeout_env_disables_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-positive env deadline disables the feature without erroring."""
    monkeypatch.setenv("AURA_BATCH_TIMEOUT_SEC", "0")
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "echo", "args": {"msg": "ok"}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_echo_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    async for _ in loop.run_turn(history=history):
        pass

    assert [m.status for m in history if isinstance(m, ToolMessage)] == ["success"]


@pytest.mark.asyncio
async def test_path_trigger_skips_non_path_and_blank_args() -> None:
    """read_file with a blank/non-str path must not feed Context.on_tool_touched."""
    def _read(path: str) -> dict[str, Any]:
        return {"content": "x", "partial": False}

    class _ReadParams(BaseModel):
        path: str

    read_tool: BaseTool = build_tool(
        name="read_file",
        description="reads",
        args_schema=_ReadParams,
        func=_read,
        is_read_only=True,
    )
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": ""}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([read_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    async for _ in loop.run_turn(history=history):
        pass

    # Blank path short-circuits the path trigger; the tool still succeeds.
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "success"


@pytest.mark.asyncio
async def test_path_trigger_swallows_resolve_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Path.resolve OSError on a touched path is swallowed, not surfaced."""
    def _read(path: str) -> dict[str, Any]:
        return {"content": "x", "partial": True}

    class _ReadParams(BaseModel):
        path: str

    read_tool: BaseTool = build_tool(
        name="read_file",
        description="reads",
        args_schema=_ReadParams,
        func=_read,
        is_read_only=True,
    )

    class _ExplodingPath:
        def __init__(self, raw: str) -> None:
            self._raw = raw

        def resolve(self) -> object:
            raise OSError("resolve failed")

    # Patch only the loop module's ``Path`` reference, never the global
    # ``pathlib.Path`` that pytest's own collection/reporting relies on.
    monkeypatch.setattr("aura.application.loop.Path", _ExplodingPath)

    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "/some/file"}, "id": "tc_1"},
        ])),
        FakeTurn(message=AIMessage(content="done")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([read_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    history: list[BaseMessage] = [HumanMessage(content="go")]
    async for _ in loop.run_turn(history=history):
        pass

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].status == "success"


@pytest.mark.asyncio
async def test_abort_skips_already_answered_tool_call() -> None:
    """Mid-batch abort must skip ids already answered, synthesising only gaps."""
    hang_started = asyncio.Event()
    block = asyncio.Event()

    class _Fast(BaseModel):
        pass

    async def _fast() -> dict[str, object]:
        return {"ok": True}

    async def _hang() -> dict[str, object]:
        hang_started.set()
        await block.wait()
        return {"ok": True}

    # Not concurrency-safe → each lands in its own batch. The fast batch
    # commits its ToolMessage first; abort lands during the hang batch so
    # the synthesiser sees ``tc_fast`` already answered and skips it.
    fast_tool = build_tool(
        name="fast", description="returns now", args_schema=_Fast,
        coroutine=_fast, is_read_only=True,
    )
    hang_tool = build_tool(
        name="hang", description="blocks", args_schema=_Fast,
        coroutine=_hang, is_read_only=True,
    )
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="", tool_calls=[
            {"name": "fast", "args": {}, "id": "tc_fast"},
            {"name": "hang", "args": {}, "id": "tc_hang"},
        ])),
        FakeTurn(message=AIMessage(content="unreached")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([fast_tool, hang_tool]),
        context=make_minimal_context(), hooks=HookChain(),
    )
    abort = AbortController()
    history: list[BaseMessage] = [HumanMessage(content="go")]

    async def _drive() -> None:
        try:
            async for _ in loop.run_turn(history=history, abort=abort):
                pass
        except (AbortException, asyncio.CancelledError):
            pass

    drive = asyncio.create_task(_drive())
    await asyncio.wait_for(hang_started.wait(), timeout=1.0)
    abort.abort("user_ctrl_c")
    try:
        await asyncio.wait_for(drive, timeout=1.0)
    finally:
        block.set()

    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    by_id = {m.tool_call_id: m for m in tool_msgs}
    # Both calls answered exactly once; the fast one keeps its real result,
    # the hung one gets a single synthetic abort message (no double-append).
    assert set(by_id) == {"tc_fast", "tc_hang"}
    assert len(tool_msgs) == 2
    assert by_id["tc_fast"].status == "success"
    assert by_id["tc_hang"].status == "error"
    assert "(aborted by user)" in str(by_id["tc_hang"].content)


@pytest.mark.asyncio
async def test_length_truncation_resumes_then_finalises() -> None:
    """A length-capped reply resumes once, then a clean reply ends the turn."""
    model = FakeChatModel(turns=[
        _length_turn("partial...", "length"),
        FakeTurn(message=AIMessage(content="complete")),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    finals: list[Final] = []
    async for ev in loop.run_turn(history=[HumanMessage(content="go")]):
        if isinstance(ev, Final):
            finals.append(ev)

    assert model.ainvoke_calls == 2
    assert len(finals) == 1
    assert finals[0].message == "complete"
    assert finals[0].reason == "natural"


@pytest.mark.asyncio
async def test_length_truncation_exhausts_and_finalises_with_reason() -> None:
    """Three consecutive truncations exhaust recovery and finalise the partial."""
    model = FakeChatModel(turns=[
        _length_turn("chunk-1", "length"),
        _length_turn("chunk-2", "max_tokens"),
        _length_turn("chunk-3", "length"),
        _length_turn("chunk-4", "length"),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([]),
        context=make_minimal_context(), hooks=HookChain(),
    )

    finals: list[Final] = []
    async for ev in loop.run_turn(history=[HumanMessage(content="go")]):
        if isinstance(ev, Final):
            finals.append(ev)

    # Initial + 3 resume attempts, then give up.
    assert model.ainvoke_calls == 4
    assert len(finals) == 1
    assert finals[0].reason == "length_recovery_exhausted"
    assert finals[0].message == "chunk-4"
