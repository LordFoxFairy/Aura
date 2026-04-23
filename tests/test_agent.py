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
from aura.core.hooks import HookChain
from aura.core.llm import UnknownModelSpecError
from aura.core.memory import project_memory, rules
from aura.core.persistence.storage import SessionStorage
from aura.schemas.events import Final
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
async def test_astream_persists_user_turn_on_cancellation(tmp_path: Path) -> None:
    """G1 contract: the user's HumanMessage persists BEFORE the model call.

    Earlier contract (pre-G1) was "cancellation never persists" so the
    post-turn save was the only persistence site. That lost the user's
    input on Ctrl-C / kill mid-stream and broke ``resume`` semantics
    (claude-code's QueryEngine saves the transcript pre-invoke; Aura
    audit B2/G1 closed the gap). Now: cancellation preserves prior
    history + the user's new turn, so the next session-resume sees
    exactly what the user typed. The assistant response does NOT land
    (the turn never reached a model reply), which is the correct
    per-claude-code semantics for an interrupted round.
    """
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
    # prior (2 msgs) + user's new turn (1 msg). No AI reply — the model
    # never completed.
    assert len(after) == 3
    assert after[0].content == "prev"
    assert after[1].content == "prior"
    assert isinstance(after[2], HumanMessage)
    assert after[2].content == "new prompt"


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
    monkeypatch.setattr(llm, "create", lambda provider, name: model_b)

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
    monkeypatch.setattr(llm, "create", lambda provider, name: model_b)

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


@pytest.mark.asyncio
async def test_clear_session_wipes_read_state(tmp_path: Path) -> None:
    # Regression guard for the must-read-first invariant: Agent.clear_session
    # rebuilds Context and swaps the hook closure. If either half silently
    # regresses — state leaks across /clear or the hook keeps pointing at the
    # old Context — previously-read files would still pass the must-read gate
    # after /clear, silently weakening the invariant. Assert both halves.
    from aura.schemas.state import LoopState
    from aura.schemas.tool import ToolResult
    from aura.tools.base import build_tool

    class _PathOldNew(BaseModel):
        path: str
        old_str: str
        new_str: str

    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="done"))])
    target = tmp_path / "f.txt"
    target.write_text("body\n")
    agent._context.record_read(target)
    assert agent._context.read_status(target) == "fresh"

    agent.clear_session()

    # State wiped:
    assert agent._context.read_status(target) == "never_read"

    # And the in-chain hook (swapped to the new Context) blocks:
    edit_tool = build_tool(
        name="edit_file",
        description="edit",
        args_schema=_PathOldNew,
        func=lambda path, old_str, new_str: {"replacements": 1},
        is_destructive=True,
    )
    result = await agent._must_read_first_hook(
        tool=edit_tool,
        args={"path": str(target), "old_str": "body", "new_str": "BODY"},
        state=LoopState(),
    )
    assert isinstance(result, ToolResult)
    assert result.ok is False
    assert result.error is not None
    assert "has not been read" in result.error


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
    monkeypatch.setattr(llm, "create", lambda provider, name: fake_model)

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
        llm, "create",
        lambda provider, name: fake_model,
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


# NOTE: test_astream_max_turns_yields_graceful_final was removed when the
# pre_model max_turns hook was dropped. The equivalent loop-level enforcement
# is covered by tests/test_loop_tool_dispatch.py::test_max_turns_caps_runaway_tool_loop
# (and siblings) — a redundant agent-level replica would just re-test the same
# Final(reason="max_turns") invariant through one more layer of indirection.


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
        llm, "create",
        lambda provider, name: fake,
    )

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "storage": {"path": str(tmp_path / "db")},
    })

    agent = build_agent(cfg)

    # default_hooks wires post_model (usage tracking), post_tool (result-size
    # budget), and Agent.__init__ adds pre_tool hooks (bash_safety,
    # must_read_first). No pre_model hook since max_turns moved to the loop.
    assert len(agent._hooks.post_model) >= 1
    assert len(agent._hooks.post_tool) >= 1
    assert len(agent._hooks.pre_tool) >= 2  # bash_safety + must_read_first
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
                    "active_form": "Running SMOKE",
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
async def test_clear_session_drops_session_rules_when_supplied(tmp_path: Path) -> None:
    from aura.core.permissions.rule import Rule
    from aura.core.permissions.session import SessionRuleSet

    session_rules = SessionRuleSet()
    session_rules.add(Rule(tool="bash", content="ls"))
    session_rules.add(Rule(tool="read_file", content=None))
    assert len(session_rules.rules()) == 2

    cfg = _minimal_config(enabled=[])
    model = FakeChatModel(turns=[])
    agent = Agent(
        config=cfg, model=model, storage=_storage(tmp_path),
        session_rules=session_rules,
    )

    agent.clear_session()
    assert session_rules.rules() == ()
    agent.close()


def test_clear_session_is_noop_when_session_rules_not_supplied(tmp_path: Path) -> None:
    # Default path: session_rules kwarg omitted → clear_session must not raise.
    cfg = _minimal_config(enabled=[])
    model = FakeChatModel(turns=[])
    agent = Agent(config=cfg, model=model, storage=_storage(tmp_path))
    agent.clear_session()  # no session_rules to touch; must be a clean no-op
    agent.close()


def test_build_agent_threads_session_rules_into_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from aura.core import llm
    from aura.core.agent import build_agent
    from aura.core.permissions.rule import Rule
    from aura.core.permissions.session import SessionRuleSet

    fake = FakeChatModel(turns=[])
    monkeypatch.setattr(llm, "create", lambda provider, name: fake)

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "storage": {"path": str(tmp_path / "db")},
    })

    session_rules = SessionRuleSet()
    session_rules.add(Rule(tool="bash", content=None))

    agent = build_agent(cfg, session_rules=session_rules)
    assert agent._session_rules is session_rules

    agent.clear_session()
    assert session_rules.rules() == ()
    agent.close()


def test_ask_user_question_registered_even_without_asker_kwarg(tmp_path: Path) -> None:
    """No CLI / no asker passed → tool is still registered so the LLM sees
    it in the bound-tools list; the failure surfaces only if it's invoked."""
    cfg = _minimal_config(enabled=["ask_user_question"])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    assert "ask_user_question" in agent._available_tools
    assert "ask_user_question" in [t.name for t in agent._registry.tools()]
    agent.close()


@pytest.mark.asyncio
async def test_ask_user_question_without_asker_raises_ToolError_on_invoke(
    tmp_path: Path,
) -> None:
    from aura.schemas.tool import ToolError

    cfg = _minimal_config(enabled=["ask_user_question"])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    tool = agent._available_tools["ask_user_question"]
    with pytest.raises(ToolError, match="no CLI asker"):
        await tool.ainvoke({"question": "hi?"})
    agent.close()


@pytest.mark.asyncio
async def test_ask_user_question_with_injected_asker_delegates(tmp_path: Path) -> None:
    captured: list[tuple[str, list[str] | None, str | None]] = []

    async def _asker(
        question: str, options: list[str] | None, default: str | None,
    ) -> str:
        captured.append((question, options, default))
        return "stub answer"

    cfg = _minimal_config(enabled=["ask_user_question"])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        question_asker=_asker,
    )
    tool = agent._available_tools["ask_user_question"]
    out = await tool.ainvoke({"question": "ready?", "options": ["yes", "no"]})
    assert out == {"answer": "stub answer"}
    assert captured == [("ready?", ["yes", "no"], None)]
    agent.close()


def test_build_agent_threads_question_asker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from aura.core import llm
    from aura.core.agent import build_agent

    fake = FakeChatModel(turns=[])
    monkeypatch.setattr(llm, "create", lambda provider, name: fake)

    async def _asker(
        question: str, options: list[str] | None, default: str | None,
    ) -> str:
        return "from-factory"

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["ask_user_question"]},
        "storage": {"path": str(tmp_path / "db")},
    })
    agent = build_agent(cfg, question_asker=_asker)
    assert "ask_user_question" in agent._available_tools
    agent.close()


def test_ask_user_question_in_default_allow_rules() -> None:
    """Calling ``ask_user_question`` must not trigger a permission prompt —
    defeats the point. Locked via DEFAULT_ALLOW_RULES."""
    from aura.core.permissions.defaults import DEFAULT_ALLOW_RULES
    names = {r.tool for r in DEFAULT_ALLOW_RULES}
    assert "ask_user_question" in names


@pytest.mark.asyncio
async def test_bash_safety_hook_blocks_zmodload_even_with_permission_allow(
    tmp_path: Path,
) -> None:
    """Safety is a HARD FLOOR — a permission hook that would allow bash
    must not override the safety short-circuit. The bash_safety hook is
    inserted at pre_tool[0] so it runs FIRST, before any user-supplied
    permission hook."""
    from langchain_core.messages import ToolMessage

    # Permission-ish hook that would happily allow any tool call.
    async def _allow_all(
        *, tool: BaseTool, args: dict[str, Any], state: Any, **_: Any,
    ) -> Any:
        return None  # pass-through → allow

    hooks = HookChain(pre_tool=[_allow_all])

    cfg = _minimal_config(enabled=["bash"])
    model = FakeChatModel(turns=[
        FakeTurn(message=AIMessage(
            content="",
            tool_calls=[{
                "id": "tc_1",
                "name": "bash",
                "args": {"command": "zmodload zsh/system"},
            }],
        )),
        FakeTurn(message=AIMessage(content="done")),
    ])
    agent = Agent(
        config=cfg, model=model, storage=_storage(tmp_path), hooks=hooks,
    )

    events: list[Any] = []
    async for e in agent.astream("try exploit"):
        events.append(e)

    # History must record the tool_call message AND a ToolMessage with an
    # error string coming from the bash_safety hook — NOT a successful ls.
    history = agent._storage.load("default")
    tool_msgs = [m for m in history if isinstance(m, ToolMessage)]
    assert tool_msgs, "expected a ToolMessage for the blocked call"
    blob = " ".join(str(m.content) for m in tool_msgs)
    assert "bash safety blocked" in blob
    assert "zsh_dangerous_command" in blob
    agent.close()


@pytest.mark.asyncio
async def test_bash_safety_hook_installed_at_position_zero(tmp_path: Path) -> None:
    """Safety must precede any caller-supplied pre_tool hook."""
    async def _noop(
        *, tool: BaseTool, args: dict[str, Any], state: Any, **_: Any,
    ) -> Any:
        return None

    hooks = HookChain(pre_tool=[_noop])
    cfg = _minimal_config(enabled=[])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        hooks=hooks,
    )
    # The safety hook closure is stored on the agent for clear_session swap.
    assert agent._hooks.pre_tool[0] is agent._bash_safety_hook
    agent.close()


@pytest.mark.asyncio
async def test_bash_safety_hook_survives_clear_session_at_position_zero(
    tmp_path: Path,
) -> None:
    async def _noop(
        *, tool: BaseTool, args: dict[str, Any], state: Any, **_: Any,
    ) -> Any:
        return None

    hooks = HookChain(pre_tool=[_noop])
    cfg = _minimal_config(enabled=[])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
        hooks=hooks,
    )
    agent.clear_session()
    assert agent._hooks.pre_tool[0] is agent._bash_safety_hook
    # And the noop / must_read_first hooks still live somewhere in the chain.
    assert _noop in agent._hooks.pre_tool
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
                    "active_form": "Running SMOKE",
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


@pytest.mark.asyncio
async def test_agent_skills_loaded_at_init_from_cwd_and_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chdir(monkeypatch, tmp_path)
    # Two layers: user-layer skill in fake HOME, project-layer skill in cwd.
    # Directory-per-skill layout (claude-code v2.1.88 compatible) — plain
    # .md files at the top of .aura/skills/ are no longer loaded.
    home = tmp_path / "_fake_home"
    (home / ".aura" / "skills" / "user-skill").mkdir(parents=True, exist_ok=True)
    (home / ".aura" / "skills" / "user-skill" / "SKILL.md").write_text(
        "---\ndescription: from user.\n---\nUSER-BODY\n",
        encoding="utf-8",
    )
    (tmp_path / ".aura" / "skills" / "proj-skill").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".aura" / "skills" / "proj-skill" / "SKILL.md").write_text(
        "---\ndescription: from project.\n---\nPROJ-BODY\n",
        encoding="utf-8",
    )

    agent = _agent(tmp_path, turns=[])
    names = {s.name for s in agent._skill_registry.list()}
    assert names == {"user-skill", "proj-skill"}
    agent.close()


@pytest.mark.asyncio
async def test_agent_record_skill_invocation_reaches_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.core.skills.types import Skill

    _chdir(monkeypatch, tmp_path)
    agent = _agent(tmp_path, turns=[])
    skill = Skill(
        name="ping",
        description="ping desc",
        body="PING-BODY",
        source_path=tmp_path / "ping.md",
        layer="project",
    )
    agent.record_skill_invocation(skill)
    messages = agent._context.build([])
    blob = " ".join(str(m.content) for m in messages)
    assert '<skill-invoked name="ping">' in blob
    assert "PING-BODY" in blob
    agent.close()


@pytest.mark.asyncio
async def test_agent_aconnect_noop_when_no_mcp_servers(tmp_path: Path) -> None:
    """No configured MCP servers → aconnect is a silent no-op, no manager
    is instantiated, and no mcp commands are stored."""
    cfg = _minimal_config(enabled=[])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    await agent.aconnect()
    assert getattr(agent, "_mcp_manager", None) is None
    assert getattr(agent, "_mcp_commands", []) == []
    agent.close()


@pytest.mark.asyncio
async def test_agent_aconnect_registers_tools_into_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aconnect() should pull tools from MCPManager and register them into
    the agent's ToolRegistry, then rebind the loop's bound model."""
    from langchain_core.tools import StructuredTool

    from aura.config.schema import MCPServerConfig
    from aura.core.mcp import manager as manager_mod
    from aura.schemas.tool import tool_metadata

    class _McpArgs(BaseModel):
        q: str = ""

    async def _coro(q: str = "") -> dict[str, Any]:
        return {}

    fake_tool = StructuredTool(
        name="mcp__gh__search",
        description="gh search",
        args_schema=_McpArgs,
        coroutine=_coro,
        metadata=tool_metadata(is_destructive=True),
    )

    class _FakeManager:
        def __init__(self, configs: list[MCPServerConfig]) -> None:
            pass

        async def start_all(self) -> tuple[list[Any], list[Any]]:
            return [fake_tool], []

        def resources_catalogue(
            self,
        ) -> list[tuple[str, str, str, str, str | None]]:
            return []

        async def read_resource(self, uri: str) -> dict[str, Any]:
            raise NotImplementedError

        async def stop_all(self) -> None:
            return None

    # Agent does ``from aura.core.mcp import MCPManager`` at module load,
    # so the name Agent resolves is the one in the agent module namespace.
    # Patch that (and the source modules for completeness).
    from aura.core import agent as agent_mod
    monkeypatch.setattr(agent_mod, "MCPManager", _FakeManager)
    monkeypatch.setattr(manager_mod, "MCPManager", _FakeManager)
    import aura.core.mcp as mcp_pkg
    monkeypatch.setattr(mcp_pkg, "MCPManager", _FakeManager)

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "mcp_servers": [
            {"name": "gh", "command": "npx", "args": ["-y", "server"]},
        ],
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    await agent.aconnect()

    assert "mcp__gh__search" in agent._registry
    # B3: async context must use aclose() (sync close with live MCP
    # manager inside a running loop raises — see test_mcp_close_timeout).
    await agent.aclose()


def test_agent_registers_task_tools_and_tasks_store(tmp_path: Path) -> None:
    cfg = _minimal_config(enabled=["task_create", "task_output"])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )
    # Store + factory live on the agent and back the two tools.
    assert agent._tasks_store is not None
    names = [t.name for t in agent._registry.tools()]
    assert "task_create" in names
    assert "task_output" in names
    agent.close()


@pytest.mark.asyncio
async def test_agent_close_cancels_running_subagent_tasks(tmp_path: Path) -> None:
    cfg = _minimal_config(enabled=["task_create"])
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=_storage(tmp_path),
    )

    # Create a pending future that never resolves, and register it into the
    # running-tasks map through the same dict the tool uses. Simulates an
    # in-flight subagent.
    loop = asyncio.get_running_loop()
    hung = loop.create_task(asyncio.sleep(10))
    agent._running_tasks["phantom"] = hung
    # Also mark the store so close() sees a record in status=running.
    rec = agent._tasks_store.create(description="phantom", prompt="hang")
    agent._running_tasks[rec.id] = hung

    agent.close()
    # Give the cancellation a tick to propagate.
    await asyncio.sleep(0)
    assert hung.cancelled() or hung.done()


@pytest.mark.asyncio
async def test_agent_aconnect_graceful_on_manager_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If MCPManager.start_all blows up entirely, aconnect must not re-raise —
    the agent starts without MCP tools."""
    from aura.config.schema import MCPServerConfig
    from aura.core.mcp import manager as manager_mod

    class _BrokenManager:
        def __init__(self, configs: list[MCPServerConfig]) -> None:
            pass

        async def start_all(self) -> tuple[list[Any], list[Any]]:
            raise RuntimeError("total failure")

        async def stop_all(self) -> None:
            return None

    from aura.core import agent as agent_mod
    monkeypatch.setattr(agent_mod, "MCPManager", _BrokenManager)
    monkeypatch.setattr(manager_mod, "MCPManager", _BrokenManager)
    import aura.core.mcp as mcp_pkg
    monkeypatch.setattr(mcp_pkg, "MCPManager", _BrokenManager)

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "mcp_servers": [
            {"name": "bad", "command": "npx", "args": []},
        ],
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="still alive"))]),
        storage=_storage(tmp_path),
    )
    # Must not raise.
    await agent.aconnect()
    # Agent is still usable.
    events = await _collect(agent, "hi")
    assert any(isinstance(e, Final) for e in events)
    agent.close()


# --------------------------------------------------------------------------
# Agent.mode — permission mode mirrored for the status bar
# --------------------------------------------------------------------------
def test_agent_mode_defaults_to_default(tmp_path: Path) -> None:
    # No mode kwarg ⇒ "default". Matches Aura's 4-mode ladder default.
    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="ok"))])
    assert agent.mode == "default"


def test_agent_mode_respects_kwarg(tmp_path: Path) -> None:
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
        mode="bypass",
    )
    assert agent.mode == "bypass"


def test_build_agent_plumbs_mode_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # build_agent is the CLI's entry point — the ``mode`` kwarg must
    # survive the trip to Agent or the bottom bar will silently show
    # "default" forever regardless of --bypass-permissions.
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "storage": {"path": str(tmp_path / "s.db")},
    })

    def _fake_create(provider: Any, model_name: str) -> Any:  # noqa: ARG001
        return FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))])

    monkeypatch.setattr("aura.core.agent.llm.create", _fake_create)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    agent = build_agent(cfg, mode="accept_edits")
    try:
        assert agent.mode == "accept_edits"
    finally:
        agent.close()


# --------------------------------------------------------------------------
# Agent.disable_bypass — Finding B: programmatic kill switch for bypass
# mode. Must refuse at BOTH construction (Agent(mode="bypass")) AND at
# runtime (set_mode("bypass")). A clean AuraConfigError carries the same
# wording as the CLI-flag path so the operator sees one consistent message.
# --------------------------------------------------------------------------
def test_agent_construct_bypass_refused_when_disable_bypass_true(
    tmp_path: Path,
) -> None:
    with pytest.raises(AuraConfigError) as exc_info:
        Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
            storage=_storage(tmp_path),
            mode="bypass",
            disable_bypass=True,
        )
    assert "bypass mode is disabled" in str(exc_info.value)
    assert "disable_bypass" in str(exc_info.value)


def test_agent_construct_non_bypass_mode_unaffected_by_disable_bypass(
    tmp_path: Path,
) -> None:
    # disable_bypass=True only blocks the "bypass" mode; every other
    # mode must still construct cleanly (regression guard — a bug that
    # raised for all modes would make disable_bypass unusable in
    # practice).
    for mode in ("default", "plan", "accept_edits"):
        agent = Agent(
            config=_minimal_config(),
            model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
            storage=_storage(tmp_path),
            mode=mode,
            disable_bypass=True,
        )
        try:
            assert agent.mode == mode
        finally:
            agent.close()


def test_agent_set_mode_bypass_refused_when_disable_bypass_true(
    tmp_path: Path,
) -> None:
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
        mode="default",
        disable_bypass=True,
    )
    try:
        with pytest.raises(AuraConfigError) as exc_info:
            agent.set_mode("bypass")
        assert "bypass mode is disabled" in str(exc_info.value)
        # Mode unchanged after refusal.
        assert agent.mode == "default"
    finally:
        agent.close()


def test_agent_set_mode_non_bypass_allowed_with_disable_bypass(
    tmp_path: Path,
) -> None:
    # Regression guard: disable_bypass must not break the shift+tab
    # cycle through default/plan/accept_edits.
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
        mode="default",
        disable_bypass=True,
    )
    try:
        agent.set_mode("plan")
        assert agent.mode == "plan"
        agent.set_mode("accept_edits")
        assert agent.mode == "accept_edits"
        agent.set_mode("default")
        assert agent.mode == "default"
    finally:
        agent.close()


def test_agent_set_mode_bypass_allowed_when_disable_bypass_false(
    tmp_path: Path,
) -> None:
    # Default — disable_bypass=False — must allow runtime switch to
    # bypass (this is what the --bypass-permissions-via-set_mode path
    # relies on).
    agent = Agent(
        config=_minimal_config(),
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
        mode="default",
        disable_bypass=False,
    )
    try:
        agent.set_mode("bypass")
        assert agent.mode == "bypass"
    finally:
        agent.close()


def test_build_agent_plumbs_disable_bypass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # build_agent is the CLI's entry point — disable_bypass must make
    # it through to Agent or the CLI-level guard is the only defense.
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "storage": {"path": str(tmp_path / "s.db")},
    })

    def _fake_create(provider: Any, model_name: str) -> Any:  # noqa: ARG001
        return FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))])

    monkeypatch.setattr("aura.core.agent.llm.create", _fake_create)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with pytest.raises(AuraConfigError):
        build_agent(cfg, mode="bypass", disable_bypass=True)


# --------------------------------------------------------------------------
# Agent.context_window — override takes precedence over llm lookup
# --------------------------------------------------------------------------
def test_agent_context_window_falls_back_to_llm_lookup(tmp_path: Path) -> None:
    # No override ⇒ resolve via the llm module's static table.
    # openai:gpt-4o-mini is in the table at 128k.
    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="ok"))])
    assert agent.context_window == 128_000


def test_agent_context_window_honors_config_override(tmp_path: Path) -> None:
    # Explicit override in AuraConfig wins regardless of model — useful
    # for extended-context deployments (e.g. Claude 4.x 1M mode).
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},  # would be 128k
        "context_window": 1_000_000,
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
    )
    assert agent.context_window == 1_000_000


# --------------------------------------------------------------------------
# Agent.pinned_tokens_estimate — local char-count estimate of pinned prefix
# --------------------------------------------------------------------------
def test_agent_pinned_tokens_estimate_is_positive(tmp_path: Path) -> None:
    # The system prompt alone guarantees a positive count — even an empty
    # project has a baseline pinned prefix. If this ever drops to zero,
    # something broke in Context.build or the system prompt loader, and
    # the bottom bar would silently show no pinned channel.
    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="ok"))])
    assert agent.pinned_tokens_estimate > 0


def test_agent_pinned_tokens_estimate_is_stable_across_reads(tmp_path: Path) -> None:
    # Property is a pure getter over a field computed once at __init__;
    # must not drift across reads (no accidental recomputation that
    # depends on live state).
    agent = _agent(tmp_path, turns=[FakeTurn(AIMessage(content="ok"))])
    first = agent.pinned_tokens_estimate
    second = agent.pinned_tokens_estimate
    assert first == second


def test_agent_context_window_unknown_model_uses_generous_default(tmp_path: Path) -> None:
    # Frontier model not in the table ⇒ 512k default (not 128k) so the
    # pct-bar doesn't lie about a 1M-window model being at "40%" when it's
    # really at 10%.
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "unknown-vendor", "protocol": "openai"}],
        "router": {"default": "unknown-vendor:some-frontier-model-we-havent-tabled"},
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))]),
        storage=_storage(tmp_path),
    )
    assert agent.context_window == 512_000
