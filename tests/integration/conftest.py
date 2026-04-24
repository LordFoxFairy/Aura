"""Shared fixtures for the integration tier.

The integration tests build a *real* :class:`aura.core.agent.Agent`, with a
real :class:`SessionStorage` (in-memory sqlite), real hook chain, real tool
dispatch. Only the LLM (:class:`tests.conftest.FakeChatModel`) is faked.

Two helpers shared across the tier:

- :func:`build_integration_agent` — one-stop Agent constructor that wires a
  FakeChatModel + in-memory storage and lets the caller override tools,
  permission mode, hooks, and the question asker. Returns ``(agent, model)``.

- :class:`ScriptedAsker` — deterministic stand-in for the CLI ``QuestionAsker``.
  Pop from a queue or reply with a sticky default; records every invocation
  so tests can assert on order / count / per-question content.

``drain`` runs ``agent.astream(prompt)`` to completion and returns all
emitted events, for tests that want to inspect the full event stream.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.hooks import HookChain
from aura.core.persistence.storage import SessionStorage
from aura.core.skills.loader import clear_conditional_state
from aura.schemas.events import AgentEvent
from aura.tools.ask_user import QuestionAsker
from tests.conftest import FakeChatModel, FakeTurn


def make_integration_config(
    enabled_tools: list[str] | None = None,
) -> AuraConfig:
    """Minimal AuraConfig for integration tests.

    Defaults to a rich tool set that covers the cases we exercise: file I/O,
    shell, plan-mode controls, subagent dispatch, and the skill tool. Callers
    that need a narrower registry can pass ``enabled_tools`` explicitly.
    """
    return AuraConfig.model_validate(
        {
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {
                "enabled": enabled_tools
                if enabled_tools is not None
                else [
                    "bash",
                    "read_file",
                    "write_file",
                    "edit_file",
                    "glob",
                    "grep",
                    "skill",
                    "web_fetch",
                    "web_search",
                    "enter_plan_mode",
                    "exit_plan_mode",
                    "task_create",
                    "task_get",
                    "task_list",
                    "task_stop",
                    "task_output",
                    "ask_user_question",
                ],
            },
        }
    )


def build_integration_agent(
    tmp_path: Path,
    turns: list[FakeTurn],
    *,
    enabled_tools: list[str] | None = None,
    available_tools: dict[str, BaseTool] | None = None,
    hooks: HookChain | None = None,
    mode: str = "default",
    question_asker: QuestionAsker | None = None,
    cwd_for_skills: Path | None = None,
) -> tuple[Agent, FakeChatModel]:
    """Build a real ``Agent`` wired to a scripted FakeChatModel.

    ``cwd_for_skills`` is the working dir the Agent inspects to load skills —
    used by the skill E2E tests to point at a tmp_path-backed ``.aura/skills``
    tree. Defaults to ``Path.cwd()`` (the inherited project's own skills).
    """
    config = make_integration_config(enabled_tools)
    model = FakeChatModel(turns=turns)
    storage = SessionStorage(tmp_path / "aura-integration.db")
    # Agent.__init__ reads ``Path.cwd()`` to seed the skill loader; tests that
    # want a tmp_path-backed skill tree chdir via monkeypatch before calling
    # this helper. Doing the chdir inside the helper would leak — tests that
    # use the ``skills_cwd`` fixture pass the expected path in for assertion.
    _ = cwd_for_skills  # accepted for documentation; the chdir is the caller's job
    agent = Agent(
        config=config,
        model=model,
        storage=storage,
        hooks=hooks,
        available_tools=available_tools,
        question_asker=question_asker,
        mode=mode,
        # Keep auto-compact off in integration tests so the LLM's scripted
        # turn count is deterministic.
        auto_compact_threshold=0,
    )
    return agent, model


async def drain(agent: Agent, prompt: str) -> list[AgentEvent]:
    """Run ``agent.astream(prompt)`` to completion; return every event."""
    events: list[AgentEvent] = []
    async for event in agent.astream(prompt):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Scripted askers
# ---------------------------------------------------------------------------


@dataclass
class AskerCall:
    """One recorded invocation of a scripted asker."""

    question: str
    options: list[str] | None
    default: str | None
    started_at: float
    finished_at: float


class ScriptedAsker:
    """Test double for :data:`QuestionAsker`.

    Call signature matches the real asker — ``(question, options, default) -> str``.
    Configure via:

    - :meth:`queue_response` — pop from a FIFO queue per call.
    - :meth:`set_default` — fallback when the queue is empty.
    - :meth:`set_delay` — simulate "user takes 100ms to answer" so mutex
      serialization is observable in tests.

    All invocations are recorded on :attr:`calls` (including timing) so
    tests can assert FIFO order / concurrency discipline.
    """

    def __init__(self, *, default: str = "Yes") -> None:
        self._queue: deque[str] = deque()
        self._default: str = default
        self._delay: float = 0.0
        self._custom: Callable[[str, list[str] | None, str | None], str] | None = None
        self.calls: list[AskerCall] = []

    def queue_response(self, answer: str) -> None:
        self._queue.append(answer)

    def set_default(self, answer: str) -> None:
        self._default = answer

    def set_delay(self, seconds: float) -> None:
        self._delay = seconds

    def set_custom(
        self, fn: Callable[[str, list[str] | None, str | None], str]
    ) -> None:
        self._custom = fn

    async def __call__(
        self,
        question: str,
        options: list[str] | None,
        default: str | None,
    ) -> str:
        started = time.monotonic()
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._custom is not None:
            answer = self._custom(question, options, default)
        elif self._queue:
            answer = self._queue.popleft()
        else:
            answer = self._default
        finished = time.monotonic()
        self.calls.append(
            AskerCall(
                question=question,
                options=list(options) if options else None,
                default=default,
                started_at=started,
                finished_at=finished,
            )
        )
        return answer


# ---------------------------------------------------------------------------
# Scripted permission asker (ToolError-driven permission hook fake)
# ---------------------------------------------------------------------------


class ScriptedPermissionAsker:
    """Test double for :class:`aura.core.hooks.permission.PermissionAsker`.

    Returns an :class:`AskerResponse` per call — queue per-tool responses
    or install a sticky default. Records ``calls`` like :class:`ScriptedAsker`
    so tests can verify FIFO ordering across concurrent subagents.
    """

    def __init__(self, *, default_response: Any = None) -> None:
        self.default_response: Any = default_response
        self._queue: deque[Any] = deque()
        self._delay: float = 0.0
        self.calls: list[dict[str, Any]] = []
        self.timings: list[tuple[float, float]] = []

    def queue(self, response: Any) -> None:  # AskerResponse
        self._queue.append(response)

    def set_delay(self, seconds: float) -> None:
        self._delay = seconds

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Any,
    ) -> Any:
        started = time.monotonic()
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self.calls.append({"tool": tool.name, "args": dict(args)})
        if self._queue:
            response = self._queue.popleft()
        elif self.default_response is not None:
            response = self.default_response
        else:
            raise RuntimeError(
                "ScriptedPermissionAsker: no queued response and no default set"
            )
        finished = time.monotonic()
        self.timings.append((started, finished))
        return response


# ---------------------------------------------------------------------------
# Autouse: reset conditional-skill module state between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_conditional_skills() -> Generator[None, None, None]:
    """Module-level ``_conditional_skills`` state leaks across tests otherwise.

    The loader caches activated conditional-skill names at module scope
    (matches claude-code's process-lifetime semantics). In the test tier
    we want every test to start clean — otherwise an earlier test's
    activation shows up in a later test's ``<skills-available>``.
    """
    clear_conditional_state()
    yield
    clear_conditional_state()


# ---------------------------------------------------------------------------
# Autouse: reset the CLI prompt mutex so each test gets a fresh loop-bound Lock.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_prompt_mutex() -> Generator[None, None, None]:
    """The CLI ``prompt_mutex`` is a lazy module-global. Pytest's asyncio
    plugin creates a fresh event loop per test, so a leftover lock bound to
    the previous loop would raise ``Lock is bound to a different event loop``
    the moment any concurrent-ask test tried to use it. Reset both sides of
    the test boundary to be defensive. Scope stays integration/ because
    hoisting to root conftest surfaced order-dependent side effects in
    unit-scope tests — the module-global's blast radius is narrow in
    practice; the reset matches that narrowness.
    """
    from aura.cli._coordination import _reset_prompt_mutex_for_tests

    _reset_prompt_mutex_for_tests()
    yield
    _reset_prompt_mutex_for_tests()

