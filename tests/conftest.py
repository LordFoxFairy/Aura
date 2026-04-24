"""Shared pytest fixtures for Aura tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from aura.core.memory.context import Context
from aura.core.memory.rules import RulesBundle


def make_minimal_context(
    *,
    cwd: Path | None = None,
    system_prompt: str = "",
    primary_memory: str = "",
    rules: RulesBundle | None = None,
) -> Context:
    """Minimal Context for loop tests — empty defaults so build(history) == [Sys, *history]."""
    return Context(
        cwd=cwd or Path("/tmp"),
        system_prompt=system_prompt,
        primary_memory=primary_memory,
        rules=rules or RulesBundle(),
    )


@pytest.fixture(autouse=True)
def clear_aura_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURA_CONFIG", raising=False)


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Redirect HOME to a fresh tmp dir for every test.

    Before V12-G, the skills loader only scanned ``~/.aura/skills/`` which
    was rarely populated on dev machines. V12-G added ``~/.claude/skills/``
    as a claude-code-compat source — typically populated with superpowers +
    other real skills. Without isolation, every Agent-constructing test
    would transitively load those real skills, polluting ``Context.build``
    output and breaking position-sensitive assertions (``cleared_positions``
    in microcompact tests, the conditional-skill promotion test, etc).

    Redirect the ``HOME`` env var only — ``Path.home()`` resolves through
    ``HOME`` on POSIX, so we get isolation without having to monkeypatch
    the method directly. This matters because some tests deliberately
    monkeypatch only ``HOME`` (e.g. ``test_history_file_path_under_aura_home``)
    and expect ``Path.home()`` to reflect their override — if we overrode
    the method here, the test's env-var monkeypatch would no-op against
    our lambda and the assertion would compare against the wrong dir.

    - Skills loader sees empty user + claude-compat layers
    - History / session storage default paths go to disposable dirs
    - Tests that deliberately want real HOME can ``monkeypatch.setenv(HOME, ...)``
      within the test — pytest's monkeypatch stack runs LIFO so the inner
      override wins over this autouse fixture.
    """
    iso = tmp_path_factory.mktemp("iso-home")
    monkeypatch.setenv("HOME", str(iso))


@dataclass
class FakeTurn:
    message: AIMessage


class FakeChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, turns: list[FakeTurn] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__["_turns"] = list(turns or [])
        self.__dict__["seen_bound_tools"] = []
        self.__dict__["ainvoke_calls"] = 0

    @property
    def seen_bound_tools(self) -> list[list[Any]]:
        return self.__dict__["seen_bound_tools"]  # type: ignore[no-any-return]

    @property
    def ainvoke_calls(self) -> int:
        return self.__dict__["ainvoke_calls"]  # type: ignore[no-any-return]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **_: Any,
    ) -> Runnable[Any, AIMessage]:
        self.__dict__["seen_bound_tools"].append(list(tools))
        return self

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        turn = self._pop_turn()
        return ChatResult(generations=[ChatGeneration(message=turn.message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        raise NotImplementedError("sync path not used")

    def _pop_turn(self) -> FakeTurn:
        turns: list[FakeTurn] = self.__dict__["_turns"]
        if not turns:
            raise RuntimeError("FakeChatModel: no scripted turns left")
        return turns.pop(0)
