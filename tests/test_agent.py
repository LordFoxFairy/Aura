"""Tests for aura.core.agent.Agent and build_agent."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.agent import Agent, build_agent
from aura.core.events import Final
from aura.core.hooks import HookChain
from aura.core.llm import UnknownModelSpecError
from aura.core.memory import project_memory, rules
from aura.core.persistence.storage import SessionStorage
from aura.tools.base import build_tool
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
    def _echo(msg: str) -> dict[str, Any]:
        return {"echoed": "x"}

    class _EchoParams(BaseModel):
        msg: str

    custom: BaseTool = build_tool(
        name="custom_echo",
        description="a custom tool",
        args_schema=_EchoParams,
        func=_echo,
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
    def _call() -> dict[str, Any]:
        return {}

    class _P(BaseModel):
        pass

    t: BaseTool = build_tool(
        name="t", description="t", args_schema=_P, func=_call, is_read_only=True,
    )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["t"]},
    })

    my_tools: dict[str, BaseTool] = {"t": t}
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

    def _call() -> dict[str, Any]:
        return {}

    class _P(BaseModel):
        pass

    custom: BaseTool = build_tool(
        name="zzz",
        description="...",
        args_schema=_P,
        func=_call,
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
    from aura.core.events import Final
    from aura.core.hooks import HookChain
    from aura.core.hooks.budget import make_max_turns_hook

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


def test_agent_current_model_reads_router_default(tmp_path: Path) -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini", "opus": "openai:gpt-4o"},
        "tools": {"enabled": []},
    })
    agent = Agent(
        config=cfg, model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    assert agent.current_model == "openai:gpt-4o-mini"
    agent.close()


def test_agent_router_aliases_excludes_default(tmp_path: Path) -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini", "opus": "openai:gpt-4o"},
        "tools": {"enabled": []},
    })
    agent = Agent(
        config=cfg, model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    assert agent.router_aliases == {"opus": "openai:gpt-4o"}
    agent.close()


@pytest.mark.asyncio
async def test_agent_builds_system_prompt(tmp_path: Path) -> None:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(message=AIMessage(content="hi"))]),
        storage=_storage(tmp_path),
    )
    assert "Aura" in agent._system_prompt
    async for _ in agent.astream("hello"):
        pass
    agent.close()


@pytest.mark.asyncio
async def test_system_prompt_prepended_to_model_messages(tmp_path: Path) -> None:
    from langchain_core.messages import SystemMessage
    from langchain_core.outputs import ChatResult

    received: list[list[BaseMessage]] = []

    class _CapturingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **_: Any,
        ) -> ChatResult:
            received.append(list(messages))
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **_)

    cfg = _minimal_config(enabled=[])
    agent = Agent(
        config=cfg,
        model=_CapturingFake(turns=[FakeTurn(message=AIMessage(content="hi"))]),  # type: ignore[call-arg]
        storage=_storage(tmp_path),
    )
    async for _ in agent.astream("hello"):
        pass
    agent.close()

    assert received
    first_call_messages = received[0]
    assert len(first_call_messages) >= 2
    assert isinstance(first_call_messages[0], SystemMessage)
    assert "Aura" in first_call_messages[0].content


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


def _chdir(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """同时 chdir + monkeypatch Path.home，避免用户真实 HOME 干扰加载。"""
    monkeypatch.chdir(path)
    fake_home = path / "_fake_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    project_memory.clear_cache()
    rules.clear_cache()


@pytest.mark.asyncio
async def test_agent_loads_primary_memory_from_cwd_AURA_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chdir(monkeypatch, tmp_path)
    (tmp_path / "AURA.md").write_text("MEMO", encoding="utf-8")

    agent = _agent(tmp_path, turns=[])
    assert "MEMO" in agent._primary_memory
    agent.close()


@pytest.mark.asyncio
async def test_agent_loads_conditional_rules_from_aura_rules_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chdir(monkeypatch, tmp_path)
    rules_dir = tmp_path / ".aura" / "rules"
    rules_dir.mkdir(parents=True)
    rule_path = rules_dir / "r.md"
    rule_path.write_text(
        "---\npaths: \"**/*.py\"\n---\nRULE-BODY\n", encoding="utf-8",
    )

    agent = _agent(tmp_path, turns=[])
    sources = [r.source_path for r in agent._rules.conditional]
    assert rule_path.resolve() in sources
    agent.close()


@pytest.mark.asyncio
async def test_agent_reuses_cache_on_repeat_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chdir(monkeypatch, tmp_path)
    (tmp_path / "AURA.md").write_text("V1", encoding="utf-8")

    # First construction populates caches.
    _agent(tmp_path, turns=[]).close()

    # Mutate the on-disk file; a second construction that re-reads disk would
    # see V2, but cache hits should still return V1.
    (tmp_path / "AURA.md").write_text("V2", encoding="utf-8")

    agent2 = _agent(tmp_path, turns=[])
    assert "V1" in agent2._primary_memory
    assert "V2" not in agent2._primary_memory
    agent2.close()


@pytest.mark.asyncio
async def test_clear_session_refreshes_primary_memory_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chdir(monkeypatch, tmp_path)
    (tmp_path / "AURA.md").write_text("V1", encoding="utf-8")

    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="ok"))])
    assert "V1" in agent._primary_memory

    (tmp_path / "AURA.md").write_text("V2-FRESH", encoding="utf-8")
    agent.clear_session()

    assert "V2-FRESH" in agent._primary_memory
    assert "V1" not in agent._primary_memory
    agent.close()


@pytest.mark.asyncio
async def test_storage_does_not_persist_context_memory_human_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chdir(monkeypatch, tmp_path)
    (tmp_path / "AURA.md").write_text("eager primary", encoding="utf-8")

    storage = _storage(tmp_path)
    model = FakeChatModel(turns=[FakeTurn(AIMessage(content="hi"))])
    agent = Agent(config=_minimal_config(), model=model, storage=storage)

    async for _ in agent.astream("go"):
        pass

    saved = storage.load("default")
    # Only the user prompt + assistant reply should be persisted; context-memory
    # tags (<project-memory>, <nested-memory>, <rule>) must not leak into storage.
    for msg in saved:
        content = str(msg.content)
        assert "<project-memory>" not in content
        assert "<nested-memory" not in content
        assert "<rule" not in content
    assert any(isinstance(m, HumanMessage) and m.content == "go" for m in saved)
    assert any(isinstance(m, AIMessage) and m.content == "hi" for m in saved)
    assert not any(isinstance(m, ToolMessage) for m in saved)
    agent.close()


class _CapturingFakeChatModel(FakeChatModel):
    """FakeChatModel that records the messages list seen per _agenerate call."""

    def __init__(self, turns: list[FakeTurn] | None = None, **kwargs: Any) -> None:
        super().__init__(turns=turns, **kwargs)
        self.__dict__["seen_messages"] = []

    @property
    def seen_messages(self) -> list[list[BaseMessage]]:
        return self.__dict__["seen_messages"]  # type: ignore[no-any-return]

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **_: Any,
    ) -> Any:
        self.__dict__["seen_messages"].append(list(messages))
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **_)


def test_todo_write_registered_when_enabled(tmp_path: Path) -> None:
    cfg = _minimal_config(enabled=["todo_write"])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    assert "todo_write" in agent._available_tools
    assert "todo_write" in [t.name for t in agent._registry.tools()]
    agent.close()


@pytest.mark.asyncio
async def test_todo_write_tool_call_injects_todos_on_next_turn(tmp_path: Path) -> None:
    cfg = _minimal_config(enabled=["todo_write"])
    model = _CapturingFakeChatModel(turns=[
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[{
                "id": "tc_1",
                "name": "todo_write",
                "args": {"todos": [{
                    "content": "SMOKE",
                    "status": "pending",
                    "activeForm": "Running SMOKE",
                }]},
            }],
        )),
        FakeTurn(message=AIMessage(content="ok")),
    ])
    agent = Agent(config=cfg, model=model, storage=_storage(tmp_path))

    async for _ in agent.astream("hi"):
        pass

    # Turn 2's messages should include a <todos> HumanMessage reflecting the
    # state set by the turn-1 tool call.
    assert len(model.seen_messages) == 2
    turn2 = model.seen_messages[1]
    todo_humans = [
        m for m in turn2
        if isinstance(m, HumanMessage) and str(m.content).startswith("<todos>")
    ]
    assert len(todo_humans) == 1
    body = str(todo_humans[0].content)
    assert "SMOKE" in body
    assert "pending" in body
    assert "Running SMOKE" in body
    agent.close()


@pytest.mark.asyncio
async def test_clear_session_wipes_todos(tmp_path: Path) -> None:
    cfg = _minimal_config(enabled=["todo_write"])
    model = _CapturingFakeChatModel(turns=[
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[{
                "id": "tc_1",
                "name": "todo_write",
                "args": {"todos": [{
                    "content": "SMOKE",
                    "status": "pending",
                    "activeForm": "Running SMOKE",
                }]},
            }],
        )),
        FakeTurn(message=AIMessage(content="ok")),
        FakeTurn(message=AIMessage(content="post-clear")),
    ])
    agent = Agent(config=cfg, model=model, storage=_storage(tmp_path))

    # Turn establishing todos.
    async for _ in agent.astream("hi"):
        pass
    assert agent._state.custom.get("todos")

    # clear_session wipes custom state including todos.
    agent.clear_session()
    assert agent._state.custom.get("todos", []) == []

    # The next turn must not carry a <todos> HumanMessage.
    async for _ in agent.astream("after-clear"):
        pass

    post_clear_messages = model.seen_messages[-1]
    todo_humans = [
        m for m in post_clear_messages
        if isinstance(m, HumanMessage) and str(m.content).startswith("<todos>")
    ]
    assert todo_humans == []
    agent.close()
