"""task_create / task_output — fire-and-forget subagent tools.

These tests exercise the whole dispatch path end-to-end: the tool returns
a task_id immediately, the subagent runs detached as an asyncio.Task in the
same event loop, and task_output reflects progress. FakeChatModel produces a
single Final turn so the subagent completes in O(event-loop-tick).

Cancellation test models the "user hits Ctrl+C mid-subagent" case — the
Agent tracks the asyncio.Task handle so close()/cancel_all can propagate
CancelledError into the child, which the run_task loop turns into
status=cancelled.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.registry import SkillRegistry
from aura.core.skills.types import Skill
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError
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


def _make_factory(tmp_path: Path) -> tuple[TasksStore, SubagentFactory]:
    store = TasksStore()
    # Subagent gets a FakeChatModel with one Final turn.
    sub_model = FakeChatModel(turns=[FakeTurn(AIMessage(content="subagent-final"))])
    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: sub_model,
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    return store, factory


@pytest.mark.asyncio
async def test_task_create_returns_running_id(tmp_path: Path) -> None:
    store, factory = _make_factory(tmp_path)
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
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
    tool = TaskCreate(store=store, factory=factory, running=tracked)
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
    tc = TaskCreate(store=store, factory=factory, running=tracked)
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
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child_agent = factory.spawn("sub-prompt")
    assert len(child_agent._config.mcp_servers) == 1
    assert child_agent._config.mcp_servers[0].name == "github"
    child_agent.close()


def test_subagent_inherits_parent_skills(tmp_path: Path) -> None:
    # Parent's loaded SkillRegistry must cross to the subagent — otherwise
    # /<skill> invocations inside the subagent see nothing. Pre-loaded
    # registry is passed through Agent(pre_loaded_skills=...) so the child
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
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        parent_skills=parent_skills,
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child_agent = factory.spawn("sub-prompt")
    names = {s.name for s in child_agent._skill_registry.list()}
    assert names == {"alpha", "beta"}
    child_agent.close()


def test_subagent_still_cannot_recurse_via_task_create() -> None:
    # Recursion guard: even though subagent inherits the parent tool set,
    # task_create + task_output MUST be stripped so a subagent can't spawn
    # further subagents (dispatch not wired into child Agents in 0.5.x).
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "task_create", "task_output"]},
    })
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child_agent = factory.spawn("sub-prompt")
    enabled = child_agent._config.tools.enabled
    assert "task_create" not in enabled
    assert "task_output" not in enabled
    child_agent.close()


def test_subagent_inherits_allowed_tools_excluding_task_tools() -> None:
    # Parent has 4 tools enabled; child sees the non-task-tool subset.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "task_create", "task_output"]},
    })
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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
    tool = TaskCreate(store=store, factory=factory, running=tasks)
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
    tool = TaskCreate(store=store, factory=factory, running={})
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
    captured: list[Agent] = []

    def _cap_model_factory() -> FakeChatModel:
        return FakeChatModel(turns=[FakeTurn(AIMessage(content="done"))])

    class _ProbeFactory(SubagentFactory):
        def spawn(
            self,
            prompt: str,
            allowed_tools: list[str] | None = None,
            *,
            agent_type: str = "general-purpose",
            task_id: str | None = None,
        ) -> Agent:
            child = super().spawn(
                prompt,
                allowed_tools,
                agent_type=agent_type,
                task_id=task_id,
            )
            captured.append(child)
            return child

    factory = _ProbeFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=_cap_model_factory,
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
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
    child.close()


def test_factory_spawn_verify_appends_verdict_system_prompt() -> None:
    # Verify type's distinguishing feature: its suffix carries the strict
    # VERDICT: output contract. Must survive the factory → Agent wiring.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file", "grep", "glob", "web_fetch", "web_search"]},
    })
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    with pytest.raises(ValueError, match="requires tools"):
        factory.spawn("p", agent_type="explore")


def test_factory_spawn_general_purpose_does_not_filter_tools() -> None:
    # General-purpose sentinel → inherit all parent tools (minus the
    # recursion-guard trio). Explicit regression guard against the
    # "empty frozenset = deny-all" misread.
    parent_config = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["bash", "read_file", "write_file", "task_create"]},
    })
    factory = SubagentFactory(
        parent_config=parent_config,
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(turns=[]),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child = factory.spawn("p", agent_type="general-purpose")
    enabled = set(child._config.tools.enabled)
    # bash + read_file + write_file inherited; task_create stripped by the
    # recursion guard.
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

    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _HangingFake(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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

    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _HangingFake(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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
async def test_run_task_timeout_fires_post_subagent_hook(
    tmp_path: Path,
) -> None:
    # On wallclock trip, the parent's post_subagent chain MUST see
    # status="failed" with the timeout error string — otherwise hook
    # consumers (telemetry, UI) never learn the child stopped.
    from aura.core.hooks import HookChain

    store = TasksStore()
    captured: list[dict[str, object]] = []

    async def _capture_hook(
        *,
        task_id: str,
        status: str,
        final_text: str,
        error: str | None,
        **_: Any,
    ) -> None:
        captured.append(
            {
                "task_id": task_id,
                "status": status,
                "final_text": final_text,
                "error": error,
            }
        )

    hooks = HookChain()
    hooks.post_subagent.append(_capture_hook)

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

    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _HangingFake(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )

    rec = store.create(description="hang", prompt="hang")
    bg = asyncio.create_task(
        run_task(
            store, factory, rec.id,
            parent_hooks=hooks,
            timeout_sec=0.15,
        )
    )
    await asyncio.wait_for(bg, timeout=2.0)

    assert len(captured) == 1
    event = captured[0]
    assert event["status"] == "failed"
    assert event["task_id"] == rec.id
    assert event["error"] is not None
    assert "subagent_timeout" in str(event["error"])


@pytest.mark.asyncio
async def test_run_task_timeout_disabled_by_zero(
    tmp_path: Path,
) -> None:
    # ``timeout_sec=0`` means "no wallclock ceiling" — the escape hatch
    # for legitimate long-running agents. A subagent that completes
    # naturally must still reach ``completed`` even with the timeout
    # switched off.
    store = TasksStore()
    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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

    factory = SubagentFactory(
        parent_config=_cfg(enabled=[]),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: _HangingFake(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
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
