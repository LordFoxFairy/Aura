"""Tests for `aura -p/--print` one-shot mode."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import cli.__main__ as _main_mod
from aura.application.hooks import HookChain
from aura.domain.events import AssistantDelta, Final, ToolCallCompleted


class _FakeStorage:
    def __init__(self, order: list[str] | None = None) -> None:
        self.cleared: list[str] = []
        self._order = order

    def clear(self, session_id: str) -> None:
        self.cleared.append(session_id)
        if self._order is not None:
            self._order.append("clear")


class _PrintAgent:
    mode = "default"
    state = object()
    hooks = HookChain()

    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events or []
        self.order: list[str] = []
        self.storage = _FakeStorage(self.order)
        self.session_id = "print-test-session"
        self.cleared_session = False
        self.connected = False
        self.closed_async = False
        self.closed_sync = False

    async def astream(self, prompt: str) -> AsyncIterator[object]:
        self.last_prompt = prompt
        for event in self.events:
            yield event

    async def aconnect(self) -> None:
        self.connected = True

    async def aclose(self, *, mcp_timeout: float = 0.0) -> None:
        del mcp_timeout
        self.closed_async = True
        self.order.append("aclose")

    def close(self, *, mcp_timeout: float = 0.0) -> None:
        del mcp_timeout
        self.closed_sync = True
        self.order.append("close")

    def clear_session(self) -> None:
        self.cleared_session = True


class _FakeWatcher:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _write_min_layout(tmp_path: Path, *, settings: dict[str, object]) -> Path:
    user_aura_dir = tmp_path / ".aura"
    user_aura_dir.mkdir()
    (user_aura_dir / "config.json").write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "p1",
                        "protocol": "openai",
                        "api_key_env": "FAKE_API_KEY",
                    }
                ],
                "router": {"default": "p1:fake-model"},
            }
        )
    )
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / ".aura").mkdir()
    (project_dir / ".aura" / "settings.json").write_text(
        json.dumps({"permissions": settings}),
    )
    return project_dir


def test_make_parser_accepts_print_prompt() -> None:
    parser = _main_mod._make_parser()
    args = parser.parse_args(["-p", "hello world"])
    assert args.print_prompt == "hello world"
    assert args.subcommand is None


@pytest.mark.asyncio
async def test_run_print_mode_returns_delta_text_preferentially() -> None:
    agent = _PrintAgent(
        events=[
            AssistantDelta("hello "),
            AssistantDelta("world"),
            Final("ignored"),
        ]
    )
    assert await _main_mod._run_print_mode(agent, "say hi") == "hello world"


@pytest.mark.asyncio
async def test_run_print_mode_falls_back_to_final_message() -> None:
    agent = _PrintAgent(events=[Final("hello world")])
    assert await _main_mod._run_print_mode(agent, "say hi") == "hello world"


@pytest.mark.asyncio
async def test_run_print_mode_raises_for_permission_needed() -> None:
    agent = _PrintAgent(
        events=[
            ToolCallCompleted(
                name="bash",
                output=None,
                error="denied: user — note: print_mode_permission_required",
            )
        ]
    )
    with pytest.raises(Exception, match="permission prompt"):
        await _main_mod._run_print_mode(agent, "run bash")


def test_main_print_mode_writes_final_answer_and_cleans_ephemeral_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = _write_min_layout(tmp_path, settings={})
    agent = _PrintAgent(
        events=[
            AssistantDelta("hello"),
            AssistantDelta(" world"),
            Final("hello world"),
        ]
    )
    built: dict[str, Any] = {}

    def fake_build_agent(*_args: object, **kwargs: object) -> _PrintAgent:
        built.update(kwargs)
        return agent

    def fake_build_default_registry(*_args: object, **_kwargs: object) -> object:
        return object()

    async def fake_dispatch(
        line: str, _agent: object, _registry: object
    ) -> object:
        from aura.application.commands.types import CommandResult

        return CommandResult(handled=False, kind="noop", text="")

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura", "-p", "say hi"])
    monkeypatch.setattr(_main_mod, "build_agent", fake_build_agent)
    monkeypatch.setattr(_main_mod, "FileWatcher", _FakeWatcher)
    monkeypatch.setattr(
        _main_mod, "build_default_registry", fake_build_default_registry
    )
    monkeypatch.setattr(_main_mod, "dispatch", fake_dispatch)

    assert _main_mod.main() == 0
    out = capsys.readouterr()
    assert out.out == "hello world\n"
    assert out.err == ""
    assert built["session_id"] != "default"
    assert built["session_id"].startswith("print-")
    assert agent.storage.cleared == [agent.session_id]
    assert agent.connected is True
    assert agent.closed_async is True
    assert agent.closed_sync is True
    assert agent.order == ["clear", "aclose", "close"]


def test_main_print_mode_handles_clear_command_without_repl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = _write_min_layout(tmp_path, settings={})
    agent = _PrintAgent()

    def fake_build_agent(*_args: object, **_kwargs: object) -> _PrintAgent:
        return agent

    def fake_build_default_registry(*_args: object, **_kwargs: object) -> object:
        return object()

    async def fake_dispatch(
        line: str, _agent: object, _registry: object
    ) -> object:
        from aura.application.commands.types import CommandResult

        assert line == "/clear"
        agent.clear_session()
        return CommandResult(handled=True, kind="print", text="session cleared")

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura", "-p", "/clear"])
    monkeypatch.setattr(_main_mod, "build_agent", fake_build_agent)
    monkeypatch.setattr(_main_mod, "FileWatcher", _FakeWatcher)
    monkeypatch.setattr(_main_mod, "build_default_registry", fake_build_default_registry)
    monkeypatch.setattr(_main_mod, "dispatch", fake_dispatch)
    monkeypatch.setattr(_main_mod, "AgentSession", _PrintAgent)

    assert _main_mod.main() == 0
    out = capsys.readouterr()
    assert out.out == "session cleared\n"
    assert agent.cleared_session is True


def test_main_print_mode_reports_permission_failure_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_dir = _write_min_layout(tmp_path, settings={})
    agent = _PrintAgent(
        events=[
            ToolCallCompleted(
                name="bash",
                output=None,
                error="denied: user — note: print_mode_permission_required",
            )
        ]
    )

    def fake_build_agent(*_args: object, **_kwargs: object) -> _PrintAgent:
        return agent

    def fake_build_default_registry(*_args: object, **_kwargs: object) -> object:
        return object()

    async def fake_dispatch(
        line: str, _agent: object, _registry: object
    ) -> object:
        from aura.application.commands.types import CommandResult

        return CommandResult(handled=False, kind="noop", text="")

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura", "-p", "run bash"])
    monkeypatch.setattr(_main_mod, "build_agent", fake_build_agent)
    monkeypatch.setattr(_main_mod, "FileWatcher", _FakeWatcher)
    monkeypatch.setattr(
        _main_mod, "build_default_registry", fake_build_default_registry
    )
    monkeypatch.setattr(_main_mod, "dispatch", fake_dispatch)

    assert _main_mod.main() == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "permission prompt" in out.err.lower()
