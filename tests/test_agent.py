"""Tests for aura.core.agent.Agent + build_agent."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.agent import Agent, build_agent
from aura.core.events import Final
from aura.core.hooks import HookChain
from aura.core.llm import UnknownModelSpecError
from aura.core.storage import SessionStorage
from aura.tools.base import AuraTool, ToolResult, build_tool
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config(enabled: list[str] | None = None) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {
            "enabled": enabled if enabled is not None else ["read_file", "write_file", "bash"]
        },
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


def _agent(
    tmp_path: Path,
    turns: list[FakeTurn],
    *,
    config: AuraConfig | None = None,
    hooks: HookChain | None = None,
) -> Agent:
    cfg = config or _minimal_config()
    model = FakeChatModel(turns=turns)
    return Agent(config=cfg, model=model, storage=_storage(tmp_path), hooks=hooks)


async def _collect(agent: Agent, prompt: str) -> list[Any]:
    events = []
    async for event in agent.astream(prompt):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_astream_yields_final_and_persists_on_success(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="hello"))])
    agent = Agent(config=_minimal_config(), model=model, storage=storage)

    events = await _collect(agent, "hi")

    assert any(isinstance(e, Final) and e.message == "hello" for e in events)
    saved = storage.load("default")
    assert len(saved) == 2
    assert isinstance(saved[0], HumanMessage)
    assert saved[0].content == "hi"
    assert isinstance(saved[1], AIMessage)
    assert saved[1].content == "hello"


@pytest.mark.asyncio
async def test_astream_does_not_persist_on_cancellation(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    prior: list[BaseMessage] = [HumanMessage(content="prev"), AIMessage(content="prior")]
    storage.save("default", prior)

    class _SlowFakeChatModel(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> Any:
            await asyncio.sleep(10)
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **_)

    model = _SlowFakeChatModel()
    agent = Agent(config=_minimal_config(), model=model, storage=storage)

    async def _run() -> None:
        async for _ in agent.astream("new prompt"):
            pass

    task = asyncio.ensure_future(_run())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    after = storage.load("default")
    assert len(after) == 2
    assert after[0].content == "prev"
    assert after[1].content == "prior"


@pytest.mark.asyncio
async def test_astream_yields_cancelled_final_before_raising(tmp_path: Path) -> None:
    async def _canceller(*, history: list[BaseMessage], state: Any, **__: Any) -> None:
        raise asyncio.CancelledError()

    hooks = HookChain(pre_model=[_canceller])
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="never"))])
    storage = _storage(tmp_path)
    agent = Agent(config=_minimal_config(), model=model, storage=storage, hooks=hooks)

    events: list[Any] = []
    with pytest.raises(asyncio.CancelledError):
        async for event in agent.astream("go"):
            events.append(event)

    assert events and isinstance(events[-1], Final)
    assert events[-1].message == "(cancelled)"


@pytest.mark.asyncio
async def test_switch_model_via_router_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini", "opus": "openai:gpt-4o"},
        "tools": {"enabled": []},
    })

    model_a = FakeChatModel(turns=[FakeTurn(AIMessage(content="first"))])
    storage = _storage(tmp_path)
    agent = Agent(config=config, model=model_a, storage=storage)

    await _collect(agent, "turn1")

    model_b = FakeChatModel(turns=[FakeTurn(AIMessage(content="second"))])

    from aura.core import llm
    monkeypatch.setattr(llm.ModelFactory, "create", lambda provider, name: (model_b, "openai"))

    agent.switch_model("opus")
    await _collect(agent, "turn2")

    history = storage.load("default")
    ai_messages = [m for m in history if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    assert ai_messages[0].content == "first"
    assert ai_messages[1].content == "second"


@pytest.mark.asyncio
async def test_switch_model_via_direct_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })

    model_a = FakeChatModel(turns=[FakeTurn(AIMessage(content="first"))])
    storage = _storage(tmp_path)
    agent = Agent(config=config, model=model_a, storage=storage)

    await _collect(agent, "turn1")

    model_b = FakeChatModel(turns=[FakeTurn(AIMessage(content="second"))])

    from aura.core import llm
    monkeypatch.setattr(llm.ModelFactory, "create", lambda provider, name: (model_b, "openai"))

    agent.switch_model("openai:gpt-4o")
    await _collect(agent, "turn2")

    history = storage.load("default")
    ai_messages = [m for m in history if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    assert ai_messages[0].content == "first"
    assert ai_messages[1].content == "second"


def test_switch_model_rejects_unknown_spec(tmp_path: Path) -> None:
    agent = _agent(tmp_path, turns=[])
    with pytest.raises(UnknownModelSpecError):
        agent.switch_model("bogus")


@pytest.mark.asyncio
async def test_clear_session_wipes_history(tmp_path: Path) -> None:
    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="hello"))])
    storage = _storage(tmp_path)

    await _collect(agent, "hi")
    assert len(storage.load("default")) == 2

    agent.clear_session()
    assert storage.load("default") == []


def test_unknown_tool_name_in_config_raises_AuraConfigError(tmp_path: Path) -> None:
    config = _minimal_config(enabled=["read_file", "ghost"])
    model = FakeChatModel()
    with pytest.raises(AuraConfigError) as exc_info:
        Agent(config=config, model=model, storage=_storage(tmp_path))
    assert "ghost" in str(exc_info.value)


@pytest.mark.asyncio
async def test_state_turn_count_accumulates_across_astream_calls(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        turns=[
            FakeTurn(AIMessage(content="one")),
            FakeTurn(AIMessage(content="two")),
        ],
    )

    await _collect(agent, "first")
    assert agent.state.turn_count == 1

    await _collect(agent, "second")
    assert agent.state.turn_count == 2


@pytest.mark.asyncio
async def test_build_agent_factory_uses_modelfactory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "storage": {"path": str(tmp_path / "factory_test.db")},
    })

    fake_model = FakeChatModel(turns=[FakeTurn(AIMessage(content="factory-output"))])

    from aura.core import llm
    monkeypatch.setattr(llm.ModelFactory, "create", lambda provider, name: (fake_model, "openai"))

    agent = build_agent(config)
    assert isinstance(agent, Agent)

    events = await _collect(agent, "test")
    assert any(isinstance(e, Final) and e.message == "factory-output" for e in events)

    db_path = config.resolved_storage_path()
    saved = SessionStorage(db_path).load("default")
    assert len(saved) == 2


@pytest.mark.asyncio
async def test_agent_accepts_custom_available_tools(tmp_path: Path) -> None:
    async def _echo_call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output={"echoed": "x"})

    class _EchoParams(BaseModel):
        msg: str

    custom: AuraTool = build_tool(
        name="custom_echo",
        description="a custom tool",
        input_model=_EchoParams,
        call=_echo_call,
        is_read_only=True,
    )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["custom_echo"]},
    })

    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="done"))])
    agent = Agent(
        config=cfg,
        model=model,
        storage=_storage(tmp_path),
        available_tools={"custom_echo": custom},
    )

    async for _ in agent.astream("hi"):
        pass


@pytest.mark.asyncio
async def test_agent_available_tools_overrides_builtins(tmp_path: Path) -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file"]},
    })

    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="x"))])
    with pytest.raises(AuraConfigError) as exc_info:
        Agent(
            config=cfg,
            model=model,
            storage=_storage(tmp_path),
            available_tools={},
        )

    assert "read_file" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_agent_copies_available_tools_dict(tmp_path: Path) -> None:
    async def _call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    class _P(BaseModel):
        pass

    t: AuraTool = build_tool(
        name="t", description="t", input_model=_P, call=_call, is_read_only=True,
    )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["t"]},
    })

    my_tools: dict[str, AuraTool] = {"t": t}
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="ok"))])
    agent = Agent(
        config=cfg, model=model, storage=_storage(tmp_path),
        available_tools=my_tools,
    )

    my_tools.clear()

    async for _ in agent.astream("go"):
        pass


@pytest.mark.asyncio
async def test_agent_close_closes_storage(tmp_path: Path) -> None:
    cfg = _minimal_config()
    storage = _storage(tmp_path)
    model = FakeChatModel(turns=[])
    agent = Agent(config=cfg, model=model, storage=storage)

    agent.close()

    with pytest.raises(sqlite3.ProgrammingError):
        storage.load("default")


@pytest.mark.asyncio
async def test_agent_async_context_manager_closes_storage(tmp_path: Path) -> None:
    cfg = _minimal_config()
    storage = _storage(tmp_path)
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))])

    async with Agent(config=cfg, model=model, storage=storage) as agent:
        async for _ in agent.astream("go"):
            pass

    with pytest.raises(sqlite3.ProgrammingError):
        storage.load("default")


@pytest.mark.asyncio
async def test_agent_respects_custom_session_id(tmp_path: Path) -> None:
    cfg = _minimal_config()
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="ok"))])
    storage = _storage(tmp_path)
    agent = Agent(
        config=cfg, model=model, storage=storage, session_id="my-session",
    )

    async for _ in agent.astream("go"):
        pass

    assert storage.load("default") == []
    loaded = storage.load("my-session")
    assert len(loaded) == 2
    agent.close()


def test_build_agent_forwards_available_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from aura.core import agent as agent_mod
    from aura.core import llm

    fake_model = FakeChatModel(turns=[])
    monkeypatch.setattr(
        llm.ModelFactory, "create",
        lambda provider, name: (fake_model, provider.protocol),
    )

    async def _call(params: BaseModel) -> ToolResult:
        return ToolResult(ok=True)

    class _P(BaseModel):
        pass

    custom: AuraTool = build_tool(
        name="zzz",
        description="...",
        input_model=_P,
        call=_call,
        is_read_only=True,
    )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["zzz"]},
        "storage": {"path": str(tmp_path / "aura.db")},
    })

    agent = agent_mod.build_agent(cfg, available_tools={"zzz": custom})
    assert agent is not None


@pytest.mark.asyncio
async def test_astream_max_turns_yields_graceful_final(tmp_path: Path) -> None:
    from aura.core.budget import make_max_turns_hook
    from aura.core.events import Final
    from aura.core.hooks import HookChain

    cfg = _minimal_config(enabled=[])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(content="hi")),
    ])
    hooks = HookChain(pre_model=[make_max_turns_hook(0)])
    agent = Agent(
        config=cfg, model=model, storage=_storage(tmp_path), hooks=hooks,
    )

    events: list[Any] = []
    async for ev in agent.astream("go"):
        events.append(ev)

    finals = [e for e in events if isinstance(e, Final)]
    assert len(finals) == 1
    assert "max_turns" in finals[0].message
    agent.close()


@pytest.mark.asyncio
async def test_clear_session_also_resets_state_counters(tmp_path: Path) -> None:
    cfg = _minimal_config(enabled=[])
    model = FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))])
    agent = Agent(config=cfg, model=model, storage=_storage(tmp_path))

    async for _ in agent.astream("x"):
        pass
    assert agent.state.turn_count == 1

    agent.clear_session()
    assert agent.state.turn_count == 0
    agent.close()


def test_build_agent_uses_default_hooks_when_none_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from aura.core import llm
    from aura.core.agent import build_agent

    fake = FakeChatModel(turns=[])
    monkeypatch.setattr(
        llm.ModelFactory, "create",
        lambda provider, name: (fake, provider.protocol),
    )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "storage": {"path": str(tmp_path / "db")},
    })

    agent = build_agent(cfg)

    assert len(agent._hooks.pre_model) >= 1
    assert len(agent._hooks.post_model) >= 1
    assert len(agent._hooks.post_tool) >= 1
    agent.close()
