"""Offline tests for the desktop headless NDJSON contract."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from aura.application.hooks.permission import make_permission_hook
from aura.application.hooks.protocols import PreToolHook
from aura.application.session import AgentSession
from aura.config.loader import load_config
from aura.config.schema import AuraConfig, PermissionsConfig
from aura.domain.events import (
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.domain.permission.outcome import Outcome
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import RuleSet
from aura.infrastructure import permission_store as perm_store
from aura.infrastructure.llm import make_model_for_spec
from aura.tools.base import build_tool
from desktop.host import headless, session_service


async def _wait_for_emitted(
    emitted: list[dict[str, Any]],
    count: int = 1,
) -> None:
    async def _wait() -> None:
        while len(emitted) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait(), timeout=1)


def test_event_to_dict_matches_frontend_contract_via_canonical_adapter() -> None:
    assert headless._event_to_dict(AssistantDelta("hi")) == {
        "event": "assistant_delta",
        "text": "hi",
    }
    assert headless._event_to_dict(
        ToolCallStarted(name="read_file", input={"path": "README.md"}),
    ) == {
        "event": "tool_call_started",
        "name": "read_file",
        "input": {"path": "README.md"},
    }
    assert headless._event_to_dict(
        ToolCallStarted(
            name="read_file",
            input={"path": "README.md"},
            id="tc_read",
        ),
    ) == {
        "event": "tool_call_started",
        "id": "tc_read",
        "name": "read_file",
        "input": {"path": "README.md"},
    }
    assert headless._event_to_dict(
        ToolCallProgress(name="bash", stream="stdout", chunk="ok\n"),
    ) == {
        "event": "tool_call_progress",
        "name": "bash",
        "stream": "stdout",
        "chunk": "ok\n",
    }
    assert headless._event_to_dict(
        ToolCallProgress(
            name="bash",
            stream="stdout",
            chunk="ok\n",
            id="tc_bash",
        ),
    ) == {
        "event": "tool_call_progress",
        "id": "tc_bash",
        "name": "bash",
        "stream": "stdout",
        "chunk": "ok\n",
    }
    assert headless._event_to_dict(
        ToolCallCompleted(name="grep", output={"matches": 1}, error=None),
    ) == {
        "event": "tool_call_completed",
        "name": "grep",
        "content": {"output": {"matches": 1}, "error": False},
    }
    assert headless._event_to_dict(
        ToolCallCompleted(name="bash", output=None, error="boom"),
    ) == {
        "event": "tool_call_completed",
        "name": "bash",
        "content": {"output": "boom", "error": True},
    }
    assert headless._event_to_dict(
        ToolCallCompleted(
            name="bash",
            output=None,
            error="boom",
            id="tc_bash",
        ),
    ) == {
        "event": "tool_call_completed",
        "id": "tc_bash",
        "name": "bash",
        "content": {"output": "boom", "error": True},
    }
    assert headless._event_to_dict(Final("done")) == {
        "event": "final",
        "message": "done",
        "reason": "natural",
    }
    assert headless._event_to_dict(Final("done", reason="max_turns")) == {
        "event": "final",
        "message": "done",
        "reason": "max_turns",
    }
    assert headless._event_to_dict(object()) == {
        "event": "unknown",
        "type": "object",
    }


def test_build_aura_state_uses_numeric_defaults() -> None:
    from aura.application.loop_state import LoopSlots

    agent = SimpleNamespace(
        state=SimpleNamespace(slots=LoopSlots()),
        current_model=None,
        mode="default",
        pinned_tokens_estimate=None,
        context_window=None,
    )

    payload = headless._build_aura_state(agent, 1.25)
    assert set(payload) == {
        "event",
        "model",
        "mode",
        "cwd",
        "tokens",
        "pinned",
        "window",
        "last_turn_seconds",
    }
    assert payload["event"] == "aura_state"
    assert payload["model"] == ""
    assert payload["mode"] == "default"
    assert payload["cwd"] == str(Path.cwd())
    assert set(payload["tokens"]) == {
        "last_input",
        "last_output",
        "last_cache_read",
        "total_input",
        "total_output",
        "total_cache_read",
        "turn_count",
    }
    assert payload["tokens"] == {
        "last_input": 0,
        "last_output": 0,
        "last_cache_read": 0,
        "total_input": 0,
        "total_output": 0,
        "total_cache_read": 0,
        "turn_count": 0,
    }
    assert payload["pinned"] == 0
    assert payload["window"] == 0
    assert payload["last_turn_seconds"] == 1.25


def test_build_aura_state_preserves_typed_token_usage() -> None:
    from aura.application.loop_state import LoopSlots
    from aura.domain.state_values import TokenStats

    agent = SimpleNamespace(
        state=SimpleNamespace(slots=LoopSlots(
            token_stats=TokenStats(
                last_input_tokens=11,
                last_output_tokens=12,
                last_cache_read_tokens=13,
                total_input_tokens=21,
                total_output_tokens=22,
                total_cache_read_tokens=23,
                turn_count=3,
            ),
        )),
        current_model="openai:gpt-4o-mini",
        mode="accept_edits",
        pinned_tokens_estimate=100,
        context_window=128000,
    )

    payload = headless._build_aura_state(agent, 0.5)
    assert payload["model"] == "openai:gpt-4o-mini"
    assert payload["mode"] == "accept_edits"
    assert payload["tokens"]["last_input"] == 11
    assert payload["tokens"]["total_output"] == 22
    assert payload["pinned"] == 100
    assert payload["window"] == 128000


class _Args(BaseModel):
    value: str


def _make_tool() -> Any:
    return build_tool(
        name="demo",
        description="demo tool",
        args_schema=_Args,
        func=lambda value: value,
        is_destructive=False,
    )


@pytest.mark.asyncio
async def test_ipc_asker_invalid_choice_defaults_to_deny(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(headless, "_emit", emitted.append)
    asker = headless.IpcAsker()
    rule = Rule("demo", None)

    task = asyncio.create_task(
        asker(tool=_make_tool(), args={"value": "x"}, rule_hint=rule),
    )
    await _wait_for_emitted(emitted)

    request_id = emitted[0]["id"]
    assert asker.feed_response({
        "kind": "permission_response",
        "id": request_id,
        "choice": "bogus",
        "feedback": "nope",
    })
    response = await asyncio.wait_for(task, timeout=1)

    assert emitted[0]["event"] == "permission_request"
    assert emitted[0]["tool"] == "demo"
    assert emitted[0]["args"] == {"value": "x"}
    assert emitted[0]["rule_hint"] == "demo"
    assert emitted[0]["is_destructive"] is False
    assert response.choice == "deny"
    assert response.feedback == "nope"


@pytest.mark.asyncio
async def test_ipc_asker_accept_and_always_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(headless, "_emit", emitted.append)
    asker = headless.IpcAsker()
    rule = Rule("demo", None)

    accept_task = asyncio.create_task(
        asker(tool=_make_tool(), args={"value": "x"}, rule_hint=rule),
    )
    await _wait_for_emitted(emitted, 1)
    assert asker.feed_response({
        "id": emitted[0]["id"],
        "choice": "accept",
        "feedback": "one shot",
    })
    accept = await asyncio.wait_for(accept_task, timeout=1)
    assert accept.choice == "accept"
    assert accept.rule is None
    assert accept.feedback == "one shot"

    always_task = asyncio.create_task(
        asker(tool=_make_tool(), args={"value": "x"}, rule_hint=rule),
    )
    await _wait_for_emitted(emitted, 2)
    assert asker.feed_response({
        "id": emitted[1]["id"],
        "choice": "always",
        "feedback": "persist it",
    })
    always = await asyncio.wait_for(always_task, timeout=1)
    assert always.choice == "always"
    assert always.rule == rule
    assert always.feedback == "persist it"


@pytest.mark.asyncio
async def test_ipc_asker_uses_safe_defaults_for_unserializable_args_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(headless, "_emit", emitted.append)
    asker = headless.IpcAsker()
    rule = Rule("demo", None)
    tool = _make_tool()
    # Strip aura_metadata so meta_dict(tool) returns {} and the asker
    # falls back to its conservative ``is_destructive=True`` default.
    object.__setattr__(tool, "aura_metadata", None)
    args: dict[str, Any] = {}
    args["self"] = args

    task = asyncio.create_task(asker(tool=tool, args=args, rule_hint=rule))
    await _wait_for_emitted(emitted)

    assert emitted[0]["args"] == {"_repr": "{'self': {...}}"}
    assert emitted[0]["is_destructive"] is True
    assert asker.feed_response({"id": emitted[0]["id"], "choice": "deny"})
    response = await asyncio.wait_for(task, timeout=1)
    assert response.choice == "deny"


def test_ipc_asker_feed_response_returns_false_for_missing_or_unknown_id() -> None:
    asker = headless.IpcAsker()

    assert asker.feed_response({"choice": "accept"}) is False
    assert asker.feed_response({"id": 123, "choice": "accept"}) is False
    assert asker.feed_response({"id": "missing", "choice": "accept"}) is False


def test_feed_permission_response_emits_error_for_unknown_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(headless, "_emit", emitted.append)
    asker = headless.IpcAsker()

    assert headless._feed_permission_response(asker, {"id": "missing"}) is False
    assert emitted == [{
        "event": "error",
        "message": "no pending permission request for id='missing'",
    }]


@pytest.mark.asyncio
async def test_ipc_asker_deny_all_pending_resolves_blocked_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(headless, "_emit", emitted.append)
    asker = headless.IpcAsker()
    rule = Rule("demo", None)

    task = asyncio.create_task(
        asker(tool=_make_tool(), args={"value": "x"}, rule_hint=rule),
    )
    await _wait_for_emitted(emitted)

    assert asker.deny_all_pending(feedback="stdin_closed") == 1
    response = await asyncio.wait_for(task, timeout=1)

    assert response.choice == "deny"
    assert response.feedback == "stdin_closed"
    assert asker.feed_response({"id": emitted[0]["id"], "choice": "accept"}) is False


@pytest.mark.asyncio
async def test_run_wires_permission_deny_ask_and_disable_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    captured_hook_kwargs: dict[str, Any] = {}
    captured_agent_kwargs: dict[str, Any] = {}
    cfg = SimpleNamespace(
        router={"default": "p1:fake-model"},
        storage=SimpleNamespace(path=str(tmp_path / "sessions.jsonl")),
        tools=SimpleNamespace(enabled=["web_fetch", "write_file", "bash"]),
    )

    def fake_permission_hook(**kwargs: Any) -> object:
        captured_hook_kwargs.update(kwargs)

        async def _hook(**_kw: Any) -> object:
            raise AssertionError("permission hook should not run")

        return _hook

    from aura.application.loop_state import LoopSlots

    class FakeAgent:
        session_id = "session-1"
        current_model = "p1:fake-model"
        mode = "default"
        state = SimpleNamespace(slots=LoopSlots())
        pinned_tokens_estimate = 0
        context_window = 0

        def __init__(self, **kwargs: Any) -> None:
            captured_agent_kwargs.update(kwargs)

        async def aclose(self) -> None:
            return None

    class FakeReader:
        async def readline(self) -> bytes:
            return b""

    class FakeLoop:
        async def connect_read_pipe(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(headless, "_emit", emitted.append)
    monkeypatch.setattr(headless, "load_config", lambda: cfg)
    monkeypatch.setattr(headless, "make_model_for_spec", lambda *_args: object())
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load",
        lambda _root: PermissionsConfig(mode="default", disable_bypass=True),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_ruleset",
        lambda *_args, **_kwargs: RuleSet((Rule("web_fetch", None),)),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_deny_ruleset",
        lambda _root: RuleSet((Rule("bash", None),)),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_ask_ruleset",
        lambda _root: RuleSet((Rule("write_file", None),)),
    )
    monkeypatch.setattr(headless, "make_permission_hook", fake_permission_hook)
    monkeypatch.setattr(headless, "AgentSession", FakeAgent)
    monkeypatch.setattr("desktop.host.headless.asyncio.StreamReader", FakeReader)
    monkeypatch.setattr(
        "desktop.host.headless.asyncio.StreamReaderProtocol",
        lambda _reader: object(),
    )
    monkeypatch.setattr(
        "desktop.host.headless.asyncio.get_running_loop",
        lambda: FakeLoop(),
    )

    assert await headless._run() == 0

    assert [rule.tool for rule in captured_hook_kwargs["rules"].rules][:1] == [
        "web_fetch",
    ]
    assert [rule.tool for rule in captured_hook_kwargs["deny_rules"].rules] == [
        "bash",
    ]
    assert [rule.tool for rule in captured_hook_kwargs["ask_rules"].rules] == [
        "write_file",
    ]
    assert captured_agent_kwargs["disable_bypass"] is True
    assert emitted[-1] == {"event": "exited"}


@pytest.mark.asyncio
async def test_run_refuses_configured_bypass_when_disable_bypass_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    cfg = SimpleNamespace(
        router={"default": "p1:fake-model"},
        storage=SimpleNamespace(path=str(tmp_path / "sessions.jsonl")),
        tools=SimpleNamespace(enabled=["web_fetch", "write_file", "bash"]),
    )

    class ExplodingAgent:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("AgentSession should not be constructed")

    monkeypatch.setattr(headless, "_emit", emitted.append)
    monkeypatch.setattr(headless, "load_config", lambda: cfg)
    monkeypatch.setattr(headless, "make_model_for_spec", lambda *_args: object())
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load",
        lambda _root: PermissionsConfig(mode="bypass", disable_bypass=True),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_ruleset",
        lambda *_args, **_kwargs: RuleSet((Rule("web_fetch", None),)),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_deny_ruleset",
        lambda _root: RuleSet((Rule("bash", None),)),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_ask_ruleset",
        lambda _root: RuleSet((Rule("write_file", None),)),
    )
    monkeypatch.setattr(headless, "AgentSession", ExplodingAgent)

    assert await headless._run() == 1
    assert emitted == [{
        "event": "error",
        "message": (
            "bypass mode is disabled by config "
            "(permissions.disable_bypass=true)"
        ),
    }]


def test_desktop_session_service_module_exists() -> None:
    assert hasattr(session_service, "run_session_driver")


@pytest.mark.asyncio
async def test_headless_run_delegates_to_session_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_driver(**_: object) -> int:
        return 7

    monkeypatch.setattr(session_service, "run_session_driver", fake_driver)

    assert await headless._run() == 7


def _build_minimal_driver_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emitted: list[dict[str, Any]],
) -> None:
    """Wire up just enough monkeypatches so ``run_session_driver`` runs."""
    cfg = SimpleNamespace(
        router={"default": "p1:fake-model"},
        storage=SimpleNamespace(path=str(tmp_path / "sessions.jsonl")),
        tools=SimpleNamespace(enabled=["web_fetch"]),
    )
    monkeypatch.setattr(headless, "_emit", emitted.append)
    monkeypatch.setattr(headless, "load_config", lambda: cfg)
    monkeypatch.setattr(headless, "make_model_for_spec", lambda *_args: object())
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load",
        lambda _root: PermissionsConfig(mode="default", disable_bypass=False),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_ruleset",
        lambda *_a, **_kw: RuleSet((Rule("web_fetch", None),)),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_deny_ruleset",
        lambda _root: RuleSet(()),
    )
    monkeypatch.setattr(
        "desktop.host.headless.perm_store.load_ask_ruleset",
        lambda _root: RuleSet(()),
    )

    def _hook_factory(**_kwargs: Any) -> object:
        async def _hook(**_kw: Any) -> object:
            raise AssertionError("permission hook should not run")
        return _hook

    monkeypatch.setattr(headless, "make_permission_hook", _hook_factory)


class _ScriptedReader:
    """In-memory NDJSON reader feeding lines from a script."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


@pytest.mark.asyncio
async def test_session_driver_emits_exited_after_final_when_turn_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel-and-await ordering: cancelled turn's final reaches the wire BEFORE exited."""
    from aura.application.loop_state import LoopSlots

    turn_started = asyncio.Event()

    class FakeAgent:
        session_id = "session-1"
        current_model = "p1:fake-model"
        mode = "default"
        state = SimpleNamespace(slots=LoopSlots())
        pinned_tokens_estimate = 0
        context_window = 0

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def astream(self, _prompt: str) -> Any:
            turn_started.set()
            try:
                # Block forever until cancelled — simulates a long-running turn.
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                yield Final("(cancelled)", reason="aborted")
                raise

        def drain_protocol_events(self) -> list[dict[str, Any]]:
            return []

        @property
        def pending_protocol_events(self) -> tuple[dict[str, Any], ...]:
            return ()

        async def aclose(self) -> None:
            return None

    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)
    monkeypatch.setattr(headless, "AgentSession", FakeAgent)

    reader = _ScriptedReader([
        b'{"kind":"prompt","text":"hi"}\n',
    ])

    def emit(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    async def _run_with_cancel() -> None:
        task = asyncio.create_task(
            session_service.run_session_driver(
                emit=emit,
                reader=reader,
                load_config_fn=load_config,
                make_model_for_spec_fn=make_model_for_spec,
                make_permission_hook_fn=make_permission_hook,
                agent_cls=cast(type[AgentSession], FakeAgent),
                perm_store_module=perm_store,
            ),
        )
        # Wait until astream is actually running — otherwise cancel races
        # with the readline loop before the turn is in-flight.
        await asyncio.wait_for(turn_started.wait(), timeout=5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    await _run_with_cancel()

    kinds = [ev["event"] for ev in emitted]
    # exited is exactly one event and comes last.
    assert kinds.count("exited") == 1
    assert kinds[-1] == "exited"
    # final("(cancelled)") must precede exited (B1 fix: await turn_task after cancel).
    assert "final" in kinds
    assert kinds.index("final") < kinds.index("exited")


@pytest.mark.asyncio
async def test_session_driver_emits_exited_exactly_once_on_clean_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No KeyboardInterrupt path means a single ``exited`` per session, not two."""
    from aura.application.loop_state import LoopSlots

    class FakeAgent:
        session_id = "session-1"
        current_model = "p1:fake-model"
        mode = "default"
        state = SimpleNamespace(slots=LoopSlots())
        pinned_tokens_estimate = 0
        context_window = 0

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def drain_protocol_events(self) -> list[dict[str, Any]]:
            return []

        @property
        def pending_protocol_events(self) -> tuple[dict[str, Any], ...]:
            return ()

        async def aclose(self) -> None:
            return None

    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)
    monkeypatch.setattr(headless, "AgentSession", FakeAgent)

    # EOF immediately — clean close path through the while-loop.
    reader = _ScriptedReader([])

    def emit(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    rc = await session_service.run_session_driver(
        emit=emit,
        reader=reader,
        load_config_fn=load_config,
        make_model_for_spec_fn=make_model_for_spec,
        make_permission_hook_fn=make_permission_hook,
        agent_cls=cast(type[AgentSession], FakeAgent),
        perm_store_module=perm_store,
    )
    assert rc == 0
    kinds = [ev["event"] for ev in emitted]
    assert kinds.count("exited") == 1
    assert kinds[-1] == "exited"


class _NoOpAgent:
    """Minimal agent satisfying the driver's lifecycle + state-snapshot contract."""

    session_id = "session-1"
    current_model = "p1:fake-model"
    mode = "default"
    pinned_tokens_estimate = 0
    context_window = 0

    def __init__(self, **_kwargs: Any) -> None:
        from aura.application.loop_state import LoopSlots

        self.state = SimpleNamespace(slots=LoopSlots())

    async def astream(self, _prompt: str) -> Any:
        if False:  # pragma: no cover - turn body unused by these branch tests
            yield None

    def drain_protocol_events(self) -> list[dict[str, Any]]:
        return []

    @property
    def pending_protocol_events(self) -> tuple[dict[str, Any], ...]:
        return ()

    async def aclose(self) -> None:
        return None


def _emitter(emitted: list[dict[str, Any]]) -> session_service.EventEmitter:
    """A nominal EventEmitter; ``list.append`` is not a structural match for mypy."""

    def emit(payload: dict[str, Any]) -> None:
        emitted.append(payload)

    return emit


async def _wait_for_event(emitted: list[dict[str, Any]], event: str) -> None:
    """Yield to the loop until ``event`` appears on the emitted wire."""
    while not any(ev.get("event") == event for ev in emitted):
        await asyncio.sleep(0)


def _as_agent_cls(fake: Any) -> type[AgentSession]:
    """Bridge a fake agent class into the driver's nominal ``type[AgentSession]``."""
    bridged: type[AgentSession] = fake
    return bridged


def _const_config(cfg: Any) -> Callable[[], AuraConfig]:
    """A zero-arg loader returning a fake config typed as the driver expects."""
    bridged: AuraConfig = cfg

    def _load() -> AuraConfig:
        return bridged

    return _load


def _const_model(model: Any) -> Callable[[str, AuraConfig], BaseChatModel]:
    """A spec→model factory returning a fake model typed as the driver expects."""
    bridged: BaseChatModel = model

    def _make(_spec: str, _cfg: AuraConfig) -> BaseChatModel:
        return bridged

    return _make


def _run_driver_with(
    reader: session_service.RequestReader,
    emitted: list[dict[str, Any]],
    agent_cls: Any = _NoOpAgent,
) -> Any:
    """Invoke the driver with the offline fakes the harness already wires."""
    return session_service.run_session_driver(
        emit=_emitter(emitted),
        reader=reader,
        load_config_fn=load_config,
        make_model_for_spec_fn=make_model_for_spec,
        make_permission_hook_fn=make_permission_hook,
        agent_cls=_as_agent_cls(agent_cls),
        perm_store_module=perm_store,
    )


@pytest.mark.asyncio
async def test_session_driver_reports_bad_json_on_top_level_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed NDJSON must surface a structured error, not crash the loop."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    reader = _ScriptedReader([b"{not json}\n"])
    rc = await _run_driver_with(reader, emitted)

    assert rc == 0
    errors = [ev for ev in emitted if ev["event"] == "error"]
    assert any(ev["message"].startswith("bad request:") for ev in errors)
    assert emitted[-1] == {"event": "exited"}


@pytest.mark.asyncio
async def test_session_driver_rejects_unknown_top_level_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported request kind is refused explicitly, never silently dropped."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    reader = _ScriptedReader([b'{"kind":"teleport"}\n'])
    rc = await _run_driver_with(reader, emitted)

    assert rc == 0
    assert {
        "event": "error",
        "message": "unsupported request kind: 'teleport'",
    } in emitted


@pytest.mark.asyncio
async def test_session_driver_rejects_missing_kind_as_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request object lacking ``kind`` resolves to None and is refused."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    reader = _ScriptedReader([b'{"text":"orphan"}\n'])
    rc = await _run_driver_with(reader, emitted)

    assert rc == 0
    assert {
        "event": "error",
        "message": "unsupported request kind: None",
    } in emitted


@pytest.mark.parametrize(
    "line",
    [
        b'{"kind":"prompt","text":""}\n',
        b'{"kind":"prompt"}\n',
        b'{"kind":"prompt","text":123}\n',
        b'{"kind":"prompt","text":null}\n',
    ],
)
@pytest.mark.asyncio
async def test_session_driver_rejects_empty_or_non_string_prompt(
    line: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty/absent/non-str prompt text must never start a turn — guard the boundary."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    reader = _ScriptedReader([line])
    rc = await _run_driver_with(reader, emitted)

    assert rc == 0
    assert {"event": "error", "message": "empty prompt"} in emitted
    assert not any(ev["event"] == "final" for ev in emitted)


@pytest.mark.asyncio
async def test_session_driver_top_level_permission_response_without_pending_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray permission_response outside any turn reports no-pending, not silence."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    reader = _ScriptedReader([b'{"kind":"permission_response","id":"ghost"}\n'])
    rc = await _run_driver_with(reader, emitted)

    assert rc == 0
    assert {
        "event": "error",
        "message": "no pending permission request for id='ghost'",
    } in emitted


@pytest.mark.asyncio
async def test_session_driver_emits_error_when_router_default_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config without router['default'] must fail fast with rc=1, not boot half-open."""
    emitted: list[dict[str, Any]] = []
    cfg: Any = SimpleNamespace(
        router={},
        storage=SimpleNamespace(path=str(tmp_path / "sessions.jsonl")),
        tools=SimpleNamespace(enabled=["web_fetch"]),
    )
    reader = _ScriptedReader([])
    rc = await session_service.run_session_driver(
        emit=_emitter(emitted),
        reader=reader,
        load_config_fn=_const_config(cfg),
        make_model_for_spec_fn=make_model_for_spec,
        make_permission_hook_fn=make_permission_hook,
        agent_cls=_as_agent_cls(_NoOpAgent),
        perm_store_module=perm_store,
    )

    assert rc == 1
    assert emitted == [{
        "event": "error",
        "message": "config.router['default'] is missing — cannot start headless",
    }]


class _ExplodingPermStore:
    """A perm store whose first lookup raises — exercises the corrupt-config guard."""

    def load(self, _project_root: Path) -> PermissionsConfig:
        raise ValueError("corrupt permissions.json")

    def load_ruleset(
        self,
        _project_root: Path,
        *,
        known_tool_names: Any = None,
    ) -> RuleSet:
        raise AssertionError("load_ruleset must not run after load() raises")

    def load_deny_ruleset(self, _project_root: Path) -> RuleSet:
        raise AssertionError("unreachable")

    def load_ask_ruleset(self, _project_root: Path) -> RuleSet:
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_session_driver_corrupt_permissions_config_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throwing permission store is reported as a typed error and aborts with rc=1."""
    emitted: list[dict[str, Any]] = []
    cfg: Any = SimpleNamespace(
        router={"default": "p1:fake-model"},
        storage=SimpleNamespace(path=str(tmp_path / "sessions.jsonl")),
        tools=SimpleNamespace(enabled=["web_fetch"]),
    )
    perm_module: session_service.PermStoreModule = _ExplodingPermStore()
    reader = _ScriptedReader([])
    rc = await session_service.run_session_driver(
        emit=_emitter(emitted),
        reader=reader,
        load_config_fn=_const_config(cfg),
        make_model_for_spec_fn=_const_model(object()),
        make_permission_hook_fn=make_permission_hook,
        agent_cls=_as_agent_cls(_NoOpAgent),
        perm_store_module=perm_module,
    )

    assert rc == 1
    assert emitted == [{
        "event": "error",
        "message": "permissions config: ValueError: corrupt permissions.json",
    }]


@pytest.mark.asyncio
async def test_session_driver_drive_turn_swallows_stream_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn-body crash is reported as one error event, never poisoning the loop."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    class ExplodingTurnAgent(_NoOpAgent):
        async def astream(self, _prompt: str) -> Any:
            raise RuntimeError("turn blew up")
            if False:  # pragma: no cover - generator marker
                yield None

    reader = _ScriptedReader([b'{"kind":"prompt","text":"go"}\n'])
    rc = await _run_driver_with(reader, emitted, agent_cls=ExplodingTurnAgent)

    assert rc == 0
    assert {
        "event": "error",
        "message": "RuntimeError: turn blew up",
    } in emitted
    assert emitted[-1] == {"event": "exited"}


class _MidTurnAgent(_NoOpAgent):
    """Holds a turn open until ``release`` is set, then emits one Final."""

    release: asyncio.Event

    def __init__(self, **_kwargs: Any) -> None:
        super().__init__(**_kwargs)
        self.release = asyncio.Event()

    async def astream(self, _prompt: str) -> Any:
        await asyncio.wait_for(self.release.wait(), timeout=5)
        yield Final("done")


class _MidTurnReader:
    """Feeds mid-turn lines, then releases the turn, then signals EOF.

    The driver polls ``readline`` while the turn runs; this reader hands out
    the scripted mid-turn lines first and only frees the turn once they are
    consumed, making the mid-turn dispatch branches deterministic.
    """

    def __init__(self, lines: list[bytes], release: asyncio.Event) -> None:
        self._lines = list(lines)
        self._release = release

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        self._release.set()
        return b""


@pytest.mark.asyncio
async def test_session_driver_mid_turn_rejects_non_permission_and_bad_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """During a turn only permission_response is accepted; others/bad-JSON error out."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    release = asyncio.Event()

    class BoundAgent(_MidTurnAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.release = release

    reader = _MidTurnReader(
        [
            b'{"kind":"prompt","text":"hi"}\n',
            b'{"kind":"prompt","text":"interrupt"}\n',
            b"{bad mid turn}\n",
        ],
        release,
    )
    rc = await _run_driver_with(reader, emitted, agent_cls=BoundAgent)

    assert rc == 0
    messages = [ev["message"] for ev in emitted if ev["event"] == "error"]
    assert (
        "only permission_response accepted mid-turn; got kind='prompt'" in messages
    )
    assert any(m.startswith("bad request mid-turn:") for m in messages)
    assert any(ev["event"] == "final" for ev in emitted)
    assert emitted[-1] == {"event": "exited"}


@pytest.mark.asyncio
async def test_session_driver_mid_turn_feeds_permission_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-turn permission_response routes to the asker, not the error channel."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    release = asyncio.Event()

    class BoundAgent(_MidTurnAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.release = release

    reader = _MidTurnReader(
        [
            b'{"kind":"prompt","text":"hi"}\n',
            b'{"kind":"permission_response","id":"none-pending"}\n',
        ],
        release,
    )
    rc = await _run_driver_with(reader, emitted, agent_cls=BoundAgent)

    assert rc == 0
    messages = [ev["message"] for ev in emitted if ev["event"] == "error"]
    assert "no pending permission request for id='none-pending'" in messages
    assert not any("mid-turn" in m for m in messages)
    assert emitted[-1] == {"event": "exited"}


class _TimeoutThenEofReader:
    """Yields a prompt, raises TimeoutError once mid-turn, then EOF.

    The driver wraps each mid-turn ``readline`` in ``wait_for``; a raised
    TimeoutError lands on the ``continue`` branch with no real wall-clock wait.
    """

    def __init__(self, release: asyncio.Event) -> None:
        self._prompt_sent = False
        self._timed_out = False
        self._release = release

    async def readline(self) -> bytes:
        if not self._prompt_sent:
            self._prompt_sent = True
            return b'{"kind":"prompt","text":"hi"}\n'
        if not self._timed_out:
            self._timed_out = True
            raise TimeoutError
        self._release.set()
        return b""


@pytest.mark.asyncio
async def test_session_driver_mid_turn_readline_timeout_keeps_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A poll timeout must loop again, not abort the turn — the turn still finalizes."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    release = asyncio.Event()

    class BoundAgent(_MidTurnAgent):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.release = release

    reader = _TimeoutThenEofReader(release)
    rc = await _run_driver_with(reader, emitted, agent_cls=BoundAgent)

    assert rc == 0
    assert any(ev["event"] == "final" for ev in emitted)
    assert not any(ev["event"] == "error" for ev in emitted)
    assert emitted[-1] == {"event": "exited"}


@pytest.mark.asyncio
async def test_session_driver_mid_turn_stdin_close_denies_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EOF while a turn is blocked on a permission ask must deny it, not hang."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    captured: dict[str, Any] = {}

    class AskingAgent(_NoOpAgent):
        async def astream(self, _prompt: str) -> Any:
            asker = captured["asker"]
            response = await asker(
                tool=_make_tool(),
                args={"value": "x"},
                rule_hint=Rule("demo", None),
            )
            captured["choice"] = response.choice
            captured["feedback"] = response.feedback
            yield Final("done")

    def capturing_hook(**kwargs: Any) -> PreToolHook:
        captured["asker"] = kwargs["asker"]

        async def _hook(**_kw: Any) -> Outcome:
            raise AssertionError("hook body unused")

        return _hook

    class BlockUntilRequestedReader:
        """Hold EOF until a permission_request is on the wire, proving a pending ask."""

        def __init__(self) -> None:
            self._prompt_sent = False

        async def readline(self) -> bytes:
            if not self._prompt_sent:
                self._prompt_sent = True
                return b'{"kind":"prompt","text":"hi"}\n'
            await asyncio.wait_for(
                _wait_for_event(emitted, "permission_request"), timeout=5,
            )
            return b""

    reader = BlockUntilRequestedReader()
    rc = await session_service.run_session_driver(
        emit=_emitter(emitted),
        reader=reader,
        load_config_fn=load_config,
        make_model_for_spec_fn=make_model_for_spec,
        make_permission_hook_fn=capturing_hook,
        agent_cls=_as_agent_cls(AskingAgent),
        perm_store_module=perm_store,
    )

    assert rc == 0
    assert captured["choice"] == "deny"
    assert captured["feedback"] == "stdin_closed"
    assert emitted[-1] == {"event": "exited"}


@pytest.mark.asyncio
async def test_session_driver_mid_turn_permission_response_resolves_pending_ask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching mid-turn permission_response must resolve the blocked ask, not error."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    captured: dict[str, Any] = {}

    class AskingAgent(_NoOpAgent):
        async def astream(self, _prompt: str) -> Any:
            asker = captured["asker"]
            response = await asker(
                tool=_make_tool(),
                args={"value": "x"},
                rule_hint=Rule("demo", None),
            )
            captured["choice"] = response.choice
            captured["feedback"] = response.feedback
            yield Final("done")

    def capturing_hook(**kwargs: Any) -> PreToolHook:
        captured["asker"] = kwargs["asker"]

        async def _hook(**_kw: Any) -> Outcome:
            raise AssertionError("hook body unused")

        return _hook

    class AnswerWhenRequestedReader:
        """Feed an accept response only once the asker has a live pending request."""

        def __init__(self) -> None:
            self._prompt_sent = False
            self._answered = False

        async def readline(self) -> bytes:
            if not self._prompt_sent:
                self._prompt_sent = True
                return b'{"kind":"prompt","text":"hi"}\n'
            if not self._answered:
                self._answered = True
                await asyncio.wait_for(
                    _wait_for_event(emitted, "permission_request"), timeout=5,
                )
                req_id = next(
                    ev["id"] for ev in emitted if ev["event"] == "permission_request"
                )
                payload = {
                    "kind": "permission_response",
                    "id": req_id,
                    "choice": "accept",
                    "feedback": "ok",
                }
                return (json.dumps(payload) + "\n").encode("utf-8")
            return b""

    reader = AnswerWhenRequestedReader()
    rc = await session_service.run_session_driver(
        emit=_emitter(emitted),
        reader=reader,
        load_config_fn=load_config,
        make_model_for_spec_fn=make_model_for_spec,
        make_permission_hook_fn=capturing_hook,
        agent_cls=_as_agent_cls(AskingAgent),
        perm_store_module=perm_store,
    )

    assert rc == 0
    assert captured["choice"] == "accept"
    assert captured["feedback"] == "ok"
    assert not any(
        ev["event"] == "error" and "no pending" in ev["message"] for ev in emitted
    )
    assert emitted[-1] == {"event": "exited"}


@pytest.mark.asyncio
async def test_session_driver_cancel_mid_turn_cancels_inflight_turn_then_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the driver mid-turn must cancel the live turn and still emit exited."""
    emitted: list[dict[str, Any]] = []
    _build_minimal_driver_env(tmp_path, monkeypatch, emitted)

    turn_started = asyncio.Event()
    turn_cancelled = asyncio.Event()

    class HangingAgent(_NoOpAgent):
        async def astream(self, _prompt: str) -> Any:
            turn_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                turn_cancelled.set()
                raise
            yield Final("unreachable")

    class HangingReader:
        """Emit a prompt, then block forever so the turn stays in-flight."""

        def __init__(self) -> None:
            self._prompt_sent = False

        async def readline(self) -> bytes:
            if not self._prompt_sent:
                self._prompt_sent = True
                return b'{"kind":"prompt","text":"hi"}\n'
            await asyncio.sleep(60)
            return b""

    reader = HangingReader()
    driver = asyncio.create_task(
        _run_driver_with(reader, emitted, agent_cls=HangingAgent),
    )
    await asyncio.wait_for(turn_started.wait(), timeout=5)
    driver.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(driver, timeout=5)

    assert turn_cancelled.is_set()
    assert emitted[-1] == {"event": "exited"}
