"""Integration tests for AgentLoop to Context path-trigger propagation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.core.hooks import HookChain
from aura.core.loop import AgentLoop
from aura.core.memory.context import Context
from aura.core.memory.rules import Rule, RulesBundle
from aura.core.registry import ToolRegistry
from aura.schemas.tool import ToolResult
from aura.tools.base import build_tool
from tests.conftest import FakeChatModel, FakeTurn


class _PathParams(BaseModel):
    path: str


class _NoArgs(BaseModel):
    pass


def _ok_tool(name: str) -> BaseTool:
    def _call(path: str) -> dict[str, Any]:
        return {"ok": True, "path": path}

    return build_tool(
        name=name, description=f"{name} tool", args_schema=_PathParams,
        func=_call, is_read_only=True, is_concurrency_safe=True,
    )


def _boom_tool(name: str) -> BaseTool:
    def _call(path: str) -> dict[str, Any]:
        raise RuntimeError(f"{name} kaboom")

    return build_tool(
        name=name, description="always raises", args_schema=_PathParams,
        func=_call, is_read_only=True,
    )


def _noop_tool(name: str) -> BaseTool:
    def _call() -> dict[str, Any]:
        return {"ok": True}

    return build_tool(
        name=name, description="no path", args_schema=_NoArgs,
        func=_call, is_read_only=True,
    )


def _py_rule(tmp_path: Path) -> Rule:
    return Rule(
        source_path=tmp_path / "py.md",
        base_dir=tmp_path.resolve(),
        globs=("**/*.py",),
        content="PY-RULE-BODY",
    )


def _context_with_py_rule(tmp_path: Path) -> Context:
    return Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[_py_rule(tmp_path)]),
    )


def _tool_turn(name: str, *, args: dict[str, Any], tc_id: str = "tc_1") -> FakeTurn:
    return FakeTurn(message=AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tc_id}],
    ))


def _final_turn(text: str = "done") -> FakeTurn:
    return FakeTurn(message=AIMessage(content=text))


# --- Success triggers ---


@pytest.mark.asyncio
async def test_successful_read_file_triggers_path_matching(tmp_path: Path) -> None:
    target = tmp_path / "x.py"
    target.write_text("", encoding="utf-8")

    ctx = _context_with_py_rule(tmp_path)
    model = FakeChatModel(turns=[
        _tool_turn("read_file", args={"path": str(target)}),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_ok_tool("read_file")]),
        context=ctx, hooks=HookChain(),
    )

    history: list[BaseMessage] = []
    history.append(HumanMessage(content="go"))
    async for _ in loop.run_turn(history=history):
        pass

    matched = [r.source_path for r in ctx._matched_rules]
    assert any(p.name == "py.md" for p in matched)


@pytest.mark.asyncio
async def test_parallel_tool_calls_all_paths_propagate(tmp_path: Path) -> None:
    a_py = tmp_path / "a.py"
    b_py = tmp_path / "b.py"
    a_py.write_text("", encoding="utf-8")
    b_py.write_text("", encoding="utf-8")

    r_a = Rule(
        source_path=(tmp_path / "ra.md"), base_dir=tmp_path.resolve(),
        globs=("a.py",), content="RA",
    )
    r_b = Rule(
        source_path=(tmp_path / "rb.md"), base_dir=tmp_path.resolve(),
        globs=("b.py",), content="RB",
    )
    ctx = Context(
        cwd=tmp_path, system_prompt="SYS", primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[r_a, r_b]),
    )

    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": str(a_py)}, "id": "t1"},
                {"name": "read_file", "args": {"path": str(b_py)}, "id": "t2"},
            ],
        )),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_ok_tool("read_file")]),
        context=ctx, hooks=HookChain(),
    )

    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    matched_globs = {r.globs for r in ctx._matched_rules}
    assert ("a.py",) in matched_globs
    assert ("b.py",) in matched_globs


# --- No-trigger paths ---


@pytest.mark.asyncio
async def test_failed_tool_call_does_not_trigger(tmp_path: Path) -> None:
    target = tmp_path / "x.py"
    target.write_text("", encoding="utf-8")

    ctx = _context_with_py_rule(tmp_path)
    model = FakeChatModel(turns=[
        _tool_turn("read_file", args={"path": str(target)}),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_boom_tool("read_file")]),
        context=ctx, hooks=HookChain(),
    )

    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert ctx._matched_rules == []


@pytest.mark.asyncio
async def test_bash_tool_success_does_not_trigger(tmp_path: Path) -> None:
    ctx = _context_with_py_rule(tmp_path)
    # PATH_TRIGGER_TOOLS excludes bash by name, so the args shape is irrelevant.
    bash = _ok_tool("bash")
    model = FakeChatModel(turns=[
        _tool_turn("bash", args={"path": "anything.py"}),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([bash]), context=ctx, hooks=HookChain(),
    )
    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert ctx._matched_rules == []


@pytest.mark.asyncio
async def test_web_fetch_tool_success_does_not_trigger(tmp_path: Path) -> None:
    ctx = _context_with_py_rule(tmp_path)
    web = _ok_tool("web_fetch")
    model = FakeChatModel(turns=[
        _tool_turn("web_fetch", args={"path": "anything.py"}),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([web]), context=ctx, hooks=HookChain(),
    )
    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert ctx._matched_rules == []


@pytest.mark.asyncio
async def test_short_circuited_tool_does_not_trigger(tmp_path: Path) -> None:
    target = tmp_path / "x.py"
    target.write_text("", encoding="utf-8")

    ctx = _context_with_py_rule(tmp_path)

    async def deny(
        *, tool: BaseTool, args: dict[str, Any], state: Any, **_: object,
    ) -> ToolResult | None:
        return ToolResult(ok=False, error="denied")

    model = FakeChatModel(turns=[
        _tool_turn("read_file", args={"path": str(target)}),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_ok_tool("read_file")]),
        context=ctx, hooks=HookChain(pre_tool=[deny]),
    )

    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert ctx._matched_rules == []


# --- Rule injection reaches the next model call ---


@pytest.mark.asyncio
async def test_matched_rule_injected_into_next_model_call(tmp_path: Path) -> None:
    target = tmp_path / "x.py"
    target.write_text("", encoding="utf-8")

    ctx = _context_with_py_rule(tmp_path)

    captured: list[list[BaseMessage]] = []

    class _CapturingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> Any:
            captured.append(list(messages))
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **_,
            )

    model = _CapturingFake(turns=[  # type: ignore[call-arg]
        _tool_turn("read_file", args={"path": str(target)}),
        _final_turn(),
    ])
    loop = AgentLoop(
        model=model, registry=ToolRegistry([_ok_tool("read_file")]),
        context=ctx, hooks=HookChain(),
    )
    async for _ in loop.run_turn(history=[HumanMessage(content="go")]):
        pass

    assert len(captured) == 2
    # First model call: no rule injected yet.
    first_contents = "\n".join(str(m.content) for m in captured[0])
    assert "PY-RULE-BODY" not in first_contents
    # Second model call: rule injected after the tool touched a .py path.
    second_contents = "\n".join(str(m.content) for m in captured[1])
    assert "<rule" in second_contents
    assert "PY-RULE-BODY" in second_contents
