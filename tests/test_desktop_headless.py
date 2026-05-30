"""Offline tests for the desktop headless NDJSON contract."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from aura.application.hooks.permission import make_permission_hook
from aura.config.loader import load_config
from aura.core.agent import Agent
from aura.domain.events import (
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)
from aura.domain.permission.rule import Rule
from aura.domain.permission.session import RuleSet
from aura.infrastructure import permission_store as perm_store
from aura.infrastructure.llm import make_model_for_spec
from aura.schemas.permissions import PermissionsConfig
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
    from aura.schemas.state import LoopSlots

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
    from aura.domain.state_values import TokenStats
    from aura.schemas.state import LoopSlots

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

    from aura.schemas.state import LoopSlots

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
    monkeypatch.setattr(headless, "Agent", FakeAgent)
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
            raise AssertionError("Agent should not be constructed")

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
    monkeypatch.setattr(headless, "Agent", ExplodingAgent)

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
    async def fake_driver() -> int:
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
    from aura.schemas.state import LoopSlots

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
    monkeypatch.setattr(headless, "Agent", FakeAgent)

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
                agent_cls=cast(type[Agent], FakeAgent),
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
    from aura.schemas.state import LoopSlots

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
    monkeypatch.setattr(headless, "Agent", FakeAgent)

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
        agent_cls=cast(type[Agent], FakeAgent),
        perm_store_module=perm_store,
    )
    assert rc == 0
    kinds = [ev["event"] for ev in emitted]
    assert kinds.count("exited") == 1
    assert kinds[-1] == "exited"
