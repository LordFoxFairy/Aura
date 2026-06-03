"""task_create / task_output — fire-and-forget subagent tools.

These tests exercise the whole dispatch path end-to-end: the tool returns
a task_id immediately, the subagent runs detached as an asyncio.Task in the
same event loop, and task_output reflects progress. FakeChatModel produces a
single Final turn so the subagent completes in O(event-loop-tick).

Cancellation test models the "user hits Ctrl+C mid-subagent" case — the
AgentSession tracks the asyncio.Task handle so close()/cancel_all can propagate
CancelledError into the child, which the run_task loop turns into
status=cancelled.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from aura.application.session import AgentSession
from aura.application.tasks.run import run_task
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.config.schema import AuraConfig
from aura.domain.skill import Skill
from aura.domain.tool import ToolError
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.skills.registry import SkillRegistry
from aura.tools.task_create import TaskCreate
from aura.tools.task_output import TaskOutput
from tests.conftest import FakeChatModel, FakeTurn


def _cfg(enabled: list[str] | None = None) -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {
            "enabled": enabled if enabled is not None else ["task_create", "task_output"]
        },
    })


def _make_factory(tmp_path: Path) -> tuple[TasksStore, SubagentSpawner[AgentSession]]:
    store = TasksStore()
    # Subagent gets a FakeChatModel with one Final turn.
    sub_model = FakeChatModel(turns=[FakeTurn(AIMessage(content="subagent-final"))])
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: sub_model,
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    return store, factory


@pytest.mark.asyncio
async def test_task_create_returns_running_id(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, spawner=factory, running=tasks)
    out = await tool.ainvoke({"description": "scan", "prompt": "find TODOs"})
    assert out["description"] == "scan"
    assert out["status"] == "running"
    assert "task_id" in out
    rec = store.get(out["task_id"])
    assert rec is not None
    assert rec.description == "scan"
    # Let the detached task finish so pytest doesn't warn about pending
    # tasks. The done-callback may have already popped the handle, so keep
    # a local snapshot around if still present.
    for _ in range(10):
        t = tasks.get(out["task_id"])
        if t is None:
            break
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_task_create_spawns_task_that_completes(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    # Snapshot the handle synchronously before the done-callback can pop it.
    snapshot: dict[str, asyncio.Task[None]] = {}

    class _SnapshotDict(dict[str, asyncio.Task[None]]):
        def __setitem__(self, k: str, v: asyncio.Task[None]) -> None:
            super().__setitem__(k, v)
            snapshot[k] = v

    tracked: _SnapshotDict = _SnapshotDict()
    tool = TaskCreate(store=store, spawner=factory, running=tracked)
    out = await tool.ainvoke({"description": "d", "prompt": "p"})
    task_id: str = out["task_id"]
    # Wait for the detached task to finish.
    await asyncio.wait_for(snapshot[task_id], timeout=2.0)
    rec = store.get(task_id)
    assert rec is not None
    assert rec.status == "completed"
    assert rec.final_result == "subagent-final"
    assert tasks == {}  # tracked is what the tool uses; quieten unused-var
    del tasks


@pytest.mark.asyncio
async def test_task_output_raises_on_unknown_id(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tool = TaskOutput(store=store)
    with pytest.raises(ToolError, match="unknown task_id"):
        await tool.ainvoke({"task_id": "no-such"})


@pytest.mark.asyncio
async def test_task_output_returns_record_for_running_task(tmp_path: Path) -> None:
    # Manually stage a running record (no detached task) — probe snapshot only.
    store, _ = _make_factory(tmp_path)
    rec = store.create(description="d", prompt="p")
    tool = TaskOutput(store=store)
    out = await tool.ainvoke({"task_id": rec.id})
    assert out["task_id"] == rec.id
    assert out["status"] == "running"
    assert out["final_result"] is None
    assert out["error"] is None


@pytest.mark.asyncio
async def test_task_output_reflects_final_result_after_subagent_finishes(
    tmp_path: Path,
) -> None:
    store, factory = _make_factory(tmp_path)
    snapshot: dict[str, asyncio.Task[None]] = {}

    class _SnapshotDict(dict[str, asyncio.Task[None]]):
        def __setitem__(self, k: str, v: asyncio.Task[None]) -> None:
            super().__setitem__(k, v)
            snapshot[k] = v

    tracked: _SnapshotDict = _SnapshotDict()
    tc = TaskCreate(store=store, spawner=factory, running=tracked)
    out = await tc.ainvoke({"description": "d", "prompt": "p"})
    task_id = out["task_id"]
    await asyncio.wait_for(snapshot[task_id], timeout=2.0)
    to = TaskOutput(store=store)
    info = await to.ainvoke({"task_id": task_id})
    assert info["status"] == "completed"
    assert info["final_result"] == "subagent-final"


def test_subagent_inherits_parent_mcp_servers() -> None:
    # Parity with claude-code: subagents inherit the parent's MCP server
    # list so they can talk to the same external tools. Each subagent runs
    # its own aconnect so connections are independent (langchain-mcp-adapters
    # spawns a fresh session per get_tools call anyway), but the SERVER
    # CONFIG LIST must cross the boundary.
    mcp_entry = {
        "name": "github",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {},
        "transport": "stdio",
    }
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "mcp_servers": [mcp_entry],
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child_agent = factory.spawn("sub-prompt")
    assert len(child_agent._config.mcp_servers) == 1
    assert child_agent._config.mcp_servers[0].name == "github"
    child_agent.close()


def test_subagent_inherits_parent_skills(tmp_path: Path) -> None:
    # Parent's loaded SkillRegistry must cross to the subagent — otherwise
    # /<skill> invocations inside the subagent see nothing. Pre-loaded
    # registry is passed through AgentSession(pre_loaded_skills=...) so the child
    # doesn't re-scan the disk (cheaper + exact parity with parent).
    parent_skills = SkillRegistry([
        Skill(
            name="alpha",
            description="alpha skill",
            body="ALPHA-BODY",
            source_path=Path("/fake/alpha.md"),
            layer="user",
        ),
        Skill(
            name="beta",
            description="beta skill",
            body="BETA-BODY",
            source_path=Path("/fake/beta.md"),
            layer="project",
        ),
    ])
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            parent_skills=parent_skills,
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child_agent = factory.spawn("sub-prompt")
    names = {s.name for s in child_agent._skill_registry.list()}
    assert names == {"alpha", "beta"}
    child_agent.close()


def test_subagent_has_no_spawn_tools() -> None:
    # One-level recursion: a spawned subagent never receives task_create /
    # task_output, so it can never spawn another subagent.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "task_create", "task_output"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child_agent = factory.spawn("sub-prompt")
    enabled = child_agent._config.tools.enabled
    assert "task_create" not in enabled
    assert "task_output" not in enabled
    child_agent.close()


def _wire_fake_model_chain(child_agent: AgentSession) -> None:
    # Test ergonomics — AgentSession.__init__ builds the child's own spawner with no
    # model_factory; inject a FakeChatModel so a direct spawn works in unit tests.
    spawner = child_agent.subagent_factory
    spawner._ctx = dataclasses.replace(
        spawner._ctx,
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


def test_subagent_strips_spawn_tools_unconditionally() -> None:
    # Defense in depth: even a direct spawn from a child's own spawner strips
    # the disallowed tools — there is no depth at which they reappear.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "task_create", "task_output"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child = factory.spawn("d1")
    _wire_fake_model_chain(child)
    grand = child.subagent_factory.spawn("d2")
    assert "task_create" not in grand._config.tools.enabled
    assert "task_output" not in grand._config.tools.enabled
    child.close()
    grand.close()


def test_subagent_inherits_non_spawn_tools() -> None:
    # The child sees the parent's tool set minus the disallowed spawn tools.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "task_create", "task_output"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child_agent = factory.spawn("sub-prompt")
    assert child_agent._config.tools.enabled == ["bash", "read_file"]
    child_agent.close()


@pytest.mark.asyncio
async def test_task_create_default_agent_type_is_general_purpose(
    tmp_path: Path,
) -> None:
    # No agent_type arg → default "general-purpose" flavor, inherits all
    # parent tools and adds no system-prompt suffix.
    store, factory = _make_factory(tmp_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, spawner=factory, running=tasks)
    out = await tool.ainvoke({"description": "d", "prompt": "p"})
    rec = store.get(out["task_id"])
    assert rec is not None
    assert rec.agent_type == "general-purpose"
    assert out["agent_type"] == "general-purpose"
    # Drain the fire-and-forget task so pytest doesn't warn.
    for _ in range(10):
        if tasks.get(out["task_id"]) is None:
            break
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_task_create_unknown_agent_type_returns_tool_error(
    tmp_path: Path,
) -> None:
    store, factory = _make_factory(tmp_path)
    tool = TaskCreate(store=store, spawner=factory, running={})
    with pytest.raises(ToolError) as ei:
        await tool.ainvoke(
            {"description": "d", "prompt": "p", "agent_type": "bogus"},
        )
    msg = str(ei.value)
    # Error surfaces the full valid-name list so the LLM can self-correct.
    for name in ("general-purpose", "explore", "verify", "plan"):
        assert name in msg
    # No orphan record left behind by a failed validation.
    assert store.list() == []


@pytest.mark.asyncio
async def test_task_create_explore_restricts_child_tools(tmp_path: Path) -> None:
    # Parent has a superset of tools; explore child must end up with only
    # the read-only allowlist.
    store = TasksStore()
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {
            "enabled": [
                "bash", "read_file", "grep", "glob", "write_file",
                "web_fetch", "web_search", "task_create", "task_output",
            ],
        },
    })
    captured: list[AgentSession] = []

    def _cap_model_factory() -> FakeChatModel:
        return FakeChatModel(turns=[FakeTurn(AIMessage(content="done"))])

    class _ProbeFactory(SubagentSpawner[AgentSession]):
        def spawn(
            self,
            prompt: str,
            allowed_tools: list[str] | None = None,
            *,
            agent_type: str = "general-purpose",
            task_id: str | None = None,
            model_spec: str | None = None,
        ) -> AgentSession:
            child = super().spawn(
                prompt,
                allowed_tools,
                agent_type=agent_type,
                task_id=task_id,
                model_spec=model_spec,
            )
            captured.append(child)
            return child

    factory = _ProbeFactory(
        SpawnContext(
            build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=_cap_model_factory,
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, spawner=factory, running=tasks)
    out = await tool.ainvoke({
        "description": "scan",
        "prompt": "find TODOs",
        "agent_type": "explore",
    })
    task_id = out["task_id"]
    # Wait for spawn to happen (run_task calls it on first event-loop tick).
    for _ in range(20):
        if captured:
            break
        await asyncio.sleep(0.01)
    assert captured, "factory.spawn was never called"
    child = captured[0]
    enabled = set(child._config.tools.enabled)
    # Only read-only tools must remain; writes and shell stripped.
    assert enabled == {"read_file", "grep", "glob", "web_fetch", "web_search"}
    assert "bash" not in enabled
    assert "write_file" not in enabled
    # System prompt suffix landed on the child.
    assert "Explore" in child._system_prompt
    # Agent_type persisted on the record for later task_get / task_list.
    rec = store.get(task_id)
    assert rec is not None
    assert rec.agent_type == "explore"
    await child.aclose()


def test_factory_spawn_verify_appends_verdict_system_prompt() -> None:
    # Verify type's distinguishing feature: its suffix carries the strict
    # VERDICT: output contract. Must survive the factory → AgentSession wiring.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file", "grep", "glob", "web_fetch", "web_search"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child = factory.spawn("audit claim X", agent_type="verify")
    assert "VERDICT:" in child._system_prompt
    assert "Verify" in child._system_prompt
    child.close()


def test_factory_spawn_plan_includes_plan_mode_tools() -> None:
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {
            "enabled": [
                "read_file", "grep", "glob", "web_fetch", "web_search",
                "enter_plan_mode", "exit_plan_mode", "bash",
            ],
        },
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child = factory.spawn("plan something", agent_type="plan")
    enabled = set(child._config.tools.enabled)
    assert "enter_plan_mode" in enabled
    assert "exit_plan_mode" in enabled
    # bash is on the parent but must be stripped for plan type.
    assert "bash" not in enabled
    child.close()


def test_factory_spawn_rejects_type_requiring_missing_parent_tools() -> None:
    # If an explore subagent asks for read_file but the parent doesn't have
    # read_file enabled, the factory must REFUSE rather than hand the child
    # a broken prompt (the suffix promises tools that don't exist).
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        # Deliberately missing read_file / grep / glob / web_*.
        "tools": {"enabled": ["bash"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    with pytest.raises(ValueError, match="requires tools"):
        factory.spawn("p", agent_type="explore")


def test_factory_spawn_general_purpose_inherits_all_but_spawn_tools() -> None:
    # General-purpose sentinel → inherit all parent tools EXCEPT the disallowed
    # spawn tools (one-level recursion).
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "write_file", "task_create"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )
    child = factory.spawn("p", agent_type="general-purpose")
    enabled = set(child._config.tools.enabled)
    # task_create stripped; the rest cross the boundary.
    assert enabled == {"bash", "read_file", "write_file"}
    # No suffix added to the prompt.
    assert "Subagent context" not in child._system_prompt
    child.close()


@pytest.mark.asyncio
async def test_parent_cancel_cascades_to_subagent(tmp_path: Path) -> None:
    store = TasksStore()

    # Subagent model that hangs forever — lets us race a cancel against it.
    class _HangingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            await asyncio.sleep(10)
            raise RuntimeError("should not get here")

    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: _HangingFake(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )

    rec = store.create(description="slow", prompt="go")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    await asyncio.sleep(0.05)
    bg.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bg
    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_parent_abort_event_cascades_to_subagent(tmp_path: Path) -> None:
    # F-07-005 — parent's abort signal travels through the factory and
    # cancels the running child within ~event-loop ticks. The bg task
    # must reach the ``cancelled`` terminal state without waiting for the
    # natural timeout / completion.
    store = TasksStore()

    class _HangingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            await asyncio.sleep(30)
            raise RuntimeError("unreachable")

    parent_abort = asyncio.Event()
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: _HangingFake(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
            parent_abort_event=parent_abort,
        )
    )

    rec = store.create(description="cascade", prompt="hang")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    # Let the child enter astream so the cascade has something to interrupt.
    await asyncio.sleep(0.05)
    parent_abort.set()
    # Bound the wait at 2s — the cascade target. CancelledError propagates
    # out of run_task because the abort watcher cancels the current_task.
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(bg, timeout=2.0)

    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled"


@pytest.mark.asyncio
async def test_subagent_spawner_does_not_propagate_abort_event_to_children() -> None:
    # One-level recursion removed the spawn chain: a child's spawner is
    # independent and does NOT carry the root's abort_event (cascade now rides
    # the per-spawn fork AbortController, not a propagated Event).
    parent_abort = asyncio.Event()
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "task_create", "task_output"]},
    })
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=parent_config,
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(turns=[]),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
            parent_abort_event=parent_abort,
        )
    )
    assert factory.abort_event is parent_abort
    child = factory.spawn("d1")
    assert child.subagent_factory.abort_event is None
    child.close()


@pytest.mark.asyncio
async def test_cancelled_subagent_is_marked_cancelled_not_completed(
    tmp_path: Path,
) -> None:
    # User-facing Ctrl+C path: a subagent whose wrapping asyncio.Task is
    # cancelled must reach mark_cancelled, never mark_completed.
    store = TasksStore()

    class _HangingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            await asyncio.sleep(30)
            raise RuntimeError("unreachable")

    captured: dict[str, Any] = {}
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: _HangingFake(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
            register_abort=lambda key, ctrl: captured.__setitem__(key, ctrl),
        )
    )

    rec = store.create(description="cancel-me", prompt="hang")
    bg = asyncio.create_task(run_task(store, factory, rec.id))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not captured:
        await asyncio.sleep(0.02)
    assert captured, "spawner must register the inherited controller via DI"

    bg.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bg

    r = store.get(rec.id)
    assert r is not None
    assert r.status == "cancelled", (
        f"cancelled subagent must be mark_cancelled, not {r.status!r}"
    )


@pytest.mark.asyncio
async def test_inherited_abort_keeps_subagent_terminal_not_completed(
    tmp_path: Path,
) -> None:
    # Load-bearing guard for the DI signal that replaced
    # object.__setattr__(child, "_inherited_abort", ...): when a subagent's
    # INHERITED AbortController fires, astream MUST re-raise (because
    # self._parent_abort is not None) so run_local_agent lands on a terminal
    # error status. If that re-raise signal is lost the child silently
    # yields Final + returns normally and gets mark_completed — the exact
    # regression this test fails on.
    store = TasksStore()

    captured: dict[str, Any] = {}

    def _register(key: str, ctrl: Any) -> None:
        captured[key] = ctrl
        # Pre-abort before the child's first turn gate so astream raises
        # AbortException deterministically (no hung model, no flake).
        ctrl.abort("user_ctrl_c")

    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(
                turns=[FakeTurn(message=AIMessage(content="hi"))],
            ),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
            register_abort=_register,
        )
    )

    rec = store.create(description="inherit-abort", prompt="go")
    await asyncio.wait_for(run_task(store, factory, rec.id), timeout=3.0)

    assert captured, "spawner must register the inherited controller via DI"
    r = store.get(rec.id)
    assert r is not None
    assert r.status != "completed", (
        "subagent whose inherited abort fired must NOT be mark_completed "
        f"(re-raise signal lost); got {r.status!r}"
    )
    assert r.status == "failed", (
        f"inherited-abort subagent should be terminal-failed, got {r.status!r}"
    )


# ---------------------------------------------------------------------------
# U4 — wall-clock timeout: a stalled subagent must NOT leave the record in
# ``running`` forever. Regression guard for the P0 dogfood bug where users
# had to manually ``task_stop`` a stuck child.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_wallclock_timeout_marks_failed(tmp_path: Path) -> None:
    # Script a subagent whose model call hangs forever. Without the
    # wallclock ceiling in run_task, ``await bg`` would deadlock — the
    # test relies on ``asyncio.wait_for`` to prove the ceiling fires.
    store = TasksStore()

    class _HangingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            # Long enough that the 0.2s timeout MUST fire before this
            # sleep returns (we need to hit the wallclock branch, not a
            # natural completion).
            await asyncio.sleep(30)
            raise RuntimeError("should not get here")

    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: _HangingFake(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )

    rec = store.create(description="hang", prompt="hang")
    # 0.2s ceiling — ample for the task-startup overhead (factory.spawn
    # + first model ainvoke dispatch are microseconds), tight enough
    # that the whole test completes in well under a second.
    bg = asyncio.create_task(
        run_task(store, factory, rec.id, timeout_sec=0.2)
    )
    # Outer wait_for is belt-and-braces: if the timeout branch is broken
    # this fails with a clear message in ~2s rather than hanging the
    # whole test run.
    await asyncio.wait_for(bg, timeout=2.0)

    r = store.get(rec.id)
    assert r is not None
    assert r.status == "failed", (
        f"expected failed after wallclock timeout, got {r.status!r}"
    )
    assert r.error is not None
    assert "subagent_timeout" in r.error
    # The error string surfaces the actual ceiling so operators know how
    # long the child was given before we pulled the plug.
    assert "0.2" in r.error


@pytest.mark.asyncio
async def test_run_task_timeout_disabled_by_zero(
    tmp_path: Path,
) -> None:
    # ``timeout_sec=0`` means "no wallclock ceiling" — the escape hatch
    # for legitimate long-running agents. A subagent that completes
    # naturally must still reach ``completed`` even with the timeout
    # switched off.
    store = TasksStore()
    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: FakeChatModel(
                turns=[FakeTurn(AIMessage(content="done"))]
            ),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )

    rec = store.create(description="fast", prompt="hi")
    await asyncio.wait_for(
        run_task(store, factory, rec.id, timeout_sec=0),
        timeout=2.0,
    )

    r = store.get(rec.id)
    assert r is not None
    assert r.status == "completed"
    assert r.final_result == "done"


@pytest.mark.asyncio
async def test_run_task_timeout_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AURA_SUBAGENT_TIMEOUT_SEC must feed the ceiling when no explicit
    # kwarg is passed — the env knob is what operators actually reach
    # for when tuning the default globally.
    monkeypatch.setenv("AURA_SUBAGENT_TIMEOUT_SEC", "0.12")
    store = TasksStore()

    class _HangingFake(FakeChatModel):
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **_: Any,
        ) -> ChatResult:
            await asyncio.sleep(30)
            raise RuntimeError("unreachable")

    factory = SubagentSpawner(
        SpawnContext(build_child=AgentSession,
            parent_config=_cfg(enabled=[]),
            parent_model_spec="openai:gpt-4o-mini",
            model_factory=lambda: _HangingFake(),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )

    rec = store.create(description="env-hang", prompt="hang")
    # No explicit timeout_sec — env must take over.
    await asyncio.wait_for(
        run_task(store, factory, rec.id),
        timeout=2.0,
    )

    r = store.get(rec.id)
    assert r is not None
    assert r.status == "failed"
    assert r.error is not None
    assert "0.12" in r.error
