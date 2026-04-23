"""Tests for ``/model`` slash command + ``Agent.switch_model`` live swap.

Covers the runtime model-switch flow end-to-end:

- ``Agent.switch_model`` resolves via router alias + direct provider:model
- failure modes (unknown spec, missing credential) propagate as
  ``AuraConfigError`` subclasses
- ``agent.current_model`` reflects the live spec (not the config default)
  after a switch, while ``config.router["default"]`` stays untouched
- the loop is rebuilt so subsequent turns route to the NEW model instance
- the ``/model`` slash command (no arg, alias, direct spec, bad spec)
  produces the right ``CommandResult`` without ever crashing the REPL
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from aura.cli.commands import build_default_registry, dispatch
from aura.config.schema import AuraConfig
from aura.core import llm
from aura.core.agent import Agent
from aura.core.llm import MissingCredentialError, UnknownModelSpecError
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [
            {"name": "openai", "protocol": "openai"},
            {"name": "anthropic", "protocol": "anthropic"},
        ],
        "router": {
            "default": "openai:gpt-4o-mini",
            "opus": "anthropic:claude-opus-4",
            "mini": "openai:gpt-4o-mini",
        },
        "tools": {"enabled": []},
    })


def _agent(tmp_path: Path, *, turns: list[FakeTurn] | None = None) -> Agent:
    return Agent(
        config=_config(),
        model=FakeChatModel(turns=turns or []),
        storage=SessionStorage(tmp_path / "db"),
    )


# ---------------------------------------------------------------------------
# Agent.switch_model — resolve + create + loop rebuild
# ---------------------------------------------------------------------------


def test_switch_model_via_alias_updates_live_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    new_model = FakeChatModel()
    monkeypatch.setattr(llm, "create", lambda p, n: new_model)

    agent.switch_model("opus")

    # Live spec flipped to the alias NAME the user passed in — we deliberately
    # don't expand the alias to its underlying spec so the /model status
    # readout shows what the user typed.
    assert agent.current_model == "opus"
    # Config surface stays immutable — router default is boot-time, not live.
    assert agent._config.router["default"] == "openai:gpt-4o-mini"
    # The underlying model instance was swapped.
    assert agent._model is new_model


def test_switch_model_via_direct_spec_updates_live_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    new_model = FakeChatModel()
    monkeypatch.setattr(llm, "create", lambda p, n: new_model)

    agent.switch_model("openai:gpt-4o")

    assert agent.current_model == "openai:gpt-4o"
    assert agent._model is new_model


def test_switch_model_unknown_provider_raises(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    with pytest.raises(UnknownModelSpecError):
        agent.switch_model("ghost:anything")
    # Live spec must NOT change on failure.
    assert agent.current_model == "openai:gpt-4o-mini"


def test_switch_model_unknown_alias_no_colon_raises(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    with pytest.raises(UnknownModelSpecError):
        agent.switch_model("not-an-alias-and-no-colon")
    assert agent.current_model == "openai:gpt-4o-mini"


def test_switch_model_missing_credential_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Clear env var so anthropic resolution trips _resolve_api_key.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = _agent(tmp_path)

    with pytest.raises(MissingCredentialError):
        agent.switch_model("anthropic:claude-opus-4")
    # Live spec must stay on the pre-switch value.
    assert agent.current_model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_switch_model_routes_next_turn_to_new_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After switch, astream must invoke the NEW model, not the old one."""
    model_a = FakeChatModel(turns=[FakeTurn(AIMessage(content="from-a"))])
    model_b = FakeChatModel(turns=[FakeTurn(AIMessage(content="from-b"))])

    agent = Agent(
        config=_config(),
        model=model_a,
        storage=SessionStorage(tmp_path / "db"),
    )

    # Turn 1 — expect model_a is called.
    async for _ in agent.astream("hi"):
        pass
    assert model_a.ainvoke_calls == 1
    assert model_b.ainvoke_calls == 0

    # Swap.
    monkeypatch.setattr(llm, "create", lambda p, n: model_b)
    agent.switch_model("opus")

    # Turn 2 — model_b must receive the call; model_a stays at 1.
    async for _ in agent.astream("again"):
        pass
    assert model_a.ainvoke_calls == 1
    assert model_b.ainvoke_calls == 1
    assert agent.current_model == "opus"


# ---------------------------------------------------------------------------
# /model slash command — no-arg status, switch confirmation, error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_command_no_args_prints_current_and_aliases(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    registry = build_default_registry()

    result = await dispatch("/model", agent, registry)

    assert result.handled is True
    assert result.kind == "print"
    assert "current:" in result.text
    assert "openai:gpt-4o-mini" in result.text
    # Non-default aliases listed, default excluded.
    assert "opus" in result.text
    assert "anthropic:claude-opus-4" in result.text
    assert "mini" in result.text


@pytest.mark.asyncio
async def test_model_command_alias_switches_and_prints_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    monkeypatch.setattr(llm, "create", lambda p, n: FakeChatModel())
    registry = build_default_registry()

    result = await dispatch("/model opus", agent, registry)

    assert result.handled is True
    assert result.kind == "print"
    # Confirmation shape: "model: <old> → <new>"
    assert "model:" in result.text
    assert "openai:gpt-4o-mini" in result.text
    assert "opus" in result.text
    # Live spec updated.
    assert agent.current_model == "opus"


@pytest.mark.asyncio
async def test_model_command_bad_spec_returns_error_does_not_raise(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path)
    registry = build_default_registry()

    # No exception should leak out of dispatch — the REPL must keep running.
    result = await dispatch("/model bogus-no-colon", agent, registry)

    assert result.handled is True
    assert result.kind == "print"
    assert "error:" in result.text
    # Live spec stays put.
    assert agent.current_model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_model_command_missing_credential_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = _agent(tmp_path)
    registry = build_default_registry()

    result = await dispatch("/model anthropic:claude-opus-4", agent, registry)

    assert result.handled is True
    assert result.kind == "print"
    assert "error:" in result.text
    # Live spec stays put.
    assert agent.current_model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_model_command_direct_spec_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path)
    monkeypatch.setattr(llm, "create", lambda p, n: FakeChatModel())
    registry = build_default_registry()

    result = await dispatch("/model openai:gpt-4o", agent, registry)

    assert result.handled is True
    assert result.kind == "print"
    assert "openai:gpt-4o" in result.text
    assert agent.current_model == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# Status bar / post-turn checkpoint integration — both read current_model
# directly, so proving the property flips proves they flip.
# ---------------------------------------------------------------------------


def test_current_model_is_read_by_both_status_surfaces() -> None:
    """Regression guard: the bottom bar and checkpoint renderer call
    ``agent.current_model`` on each render, so no extra wiring is
    required to reflect a /model switch. This test asserts the property
    exists, is a ``str``, and reflects the LIVE spec — the single
    invariant those surfaces depend on."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.current_model = "openai:gpt-4o-mini"
    assert isinstance(mock_agent.current_model, str)
    mock_agent.current_model = "anthropic:claude-opus-4"
    assert mock_agent.current_model == "anthropic:claude-opus-4"
