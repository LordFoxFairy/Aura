"""Round 7QS — periodic AgentSummarizer.

Pins the contract:

- The summary loop ticks at the configured interval, invokes the
  cheap summary model, and writes the response onto
  :attr:`TaskProgress.latest_summary`.
- ``interval_sec=0`` (or env override to 0) disables the loop entirely.
- Cancellation / terminal transition stops the loop cleanly.
- The summary model factory is the SAME shape ``web_fetch`` uses
  (:func:`aura.core.llm.make_summary_model_factory`) — when no
  ``summary_spec`` is configured, the factory falls back to the main
  model. When configured, the factory yields the cheap-tier instance.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core import llm
from aura.core.persistence.storage import SessionStorage
from aura.core.services.agent_summary import (
    AgentSummarizer,
    _resolve_interval,
    run_summary_loop,
)
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _make_summary_model(
    response_texts: list[str],
) -> tuple[FakeChatModel, list[list[BaseMessage]]]:
    """Build a FakeChatModel that yields a queue of summary responses.

    Returns the model + a captured-prompts list so tests can inspect
    what the summarizer actually fed the cheap model.
    """
    captured: list[list[BaseMessage]] = []
    turns = [FakeTurn(AIMessage(content=text)) for text in response_texts]
    model = FakeChatModel(turns=turns)
    # Wrap _agenerate to capture the prompts the summarizer sends.
    orig_agenerate = model._agenerate

    async def _spy(messages: list[BaseMessage], *a: Any, **kw: Any) -> Any:
        captured.append(list(messages))
        return await orig_agenerate(messages, *a, **kw)

    model._agenerate = _spy  # type: ignore[method-assign]
    return model, captured


# ---------------------------------------------------------------------------
# Interval resolution
# ---------------------------------------------------------------------------


def test_interval_override_wins() -> None:
    assert _resolve_interval(2.5) == 2.5


def test_interval_zero_disables() -> None:
    # Explicit 0 → 0 (disabled sentinel).
    assert _resolve_interval(0.0) == 0.0


def test_interval_env_var_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AURA_AGENT_SUMMARY_INTERVAL_SEC", "5.5")
    # No override; env wins over default.
    assert _resolve_interval(None) == 5.5


def test_interval_env_var_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_AGENT_SUMMARY_INTERVAL_SEC", "not-a-number")
    assert _resolve_interval(None) == 30.0  # default


# ---------------------------------------------------------------------------
# Summary loop directly (no run_task wrapper)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_disabled_when_interval_zero() -> None:
    """interval=0 → loop is a no-op; latest_summary stays None."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    model, _captured = _make_summary_model(["should not run"])
    # interval=0 short-circuits before the loop's first sleep.
    await run_summary_loop(
        task_id=rec.id,
        store=store,
        transcript_provider=lambda: [HumanMessage(content="hi")],
        summary_model_factory=lambda: model,
        interval_sec=0.0,
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.progress.latest_summary is None


@pytest.mark.asyncio
async def test_summary_runs_every_interval() -> None:
    """A loop running for ~3 ticks at 0.05s should land >=2 summary writes."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    model, captured = _make_summary_model([
        "wrote 3 lines",
        "ran tests",
        "found a bug",
    ])

    async def _scenario() -> None:
        # Run the loop in a background task; let it tick a few times,
        # then transition the record terminal so the loop exits.
        loop_task = asyncio.create_task(run_summary_loop(
            task_id=rec.id,
            store=store,
            transcript_provider=lambda: [
                HumanMessage(content="user prompt"),
                AIMessage(content="assistant doing things"),
            ],
            summary_model_factory=lambda: model,
            interval_sec=0.05,
        ))
        # Let at least 2 ticks fire.
        await asyncio.sleep(0.16)
        store.mark_completed(rec.id, "done")
        await asyncio.wait_for(loop_task, timeout=1.0)

    await _scenario()

    # At least 2 summary updates landed.
    assert len(captured) >= 2
    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.progress.latest_summary is not None
    # Latest summary is the LAST response the model returned (overwrite).
    assert refreshed.progress.latest_summary in (
        "wrote 3 lines", "ran tests", "found a bug",
    )
    assert refreshed.progress.summary_updated_at is not None


@pytest.mark.asyncio
async def test_summary_cancels_on_terminal() -> None:
    """A record going terminal mid-loop exits before the next tick."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    model, captured = _make_summary_model(["only one"])

    summarizer = AgentSummarizer(
        task_id=rec.id,
        store=store,
        transcript_provider=lambda: [HumanMessage(content="hi")],
        summary_model_factory=lambda: model,
        interval_sec=0.05,
    )
    summarizer.start()
    # Wait for at least one tick.
    await asyncio.sleep(0.06)
    # Mark terminal — next iteration of the loop should observe this.
    store.mark_completed(rec.id, "done")
    await asyncio.wait_for(asyncio.shield(_wait_done(summarizer)), timeout=1.0)
    assert summarizer.done


async def _wait_done(s: AgentSummarizer) -> None:
    # Spin briefly until the summarizer's task is done.
    for _ in range(100):
        if s.done:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_summary_stop_cancels_running_loop() -> None:
    """Explicit ``stop()`` collapses an in-flight loop immediately."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")
    model, _captured = _make_summary_model(["one", "two"])

    summarizer = AgentSummarizer(
        task_id=rec.id,
        store=store,
        transcript_provider=lambda: [HumanMessage(content="hi")],
        summary_model_factory=lambda: model,
        interval_sec=10.0,  # long — would block forever without cancel
    )
    summarizer.start()
    await asyncio.sleep(0.01)
    await summarizer.stop()
    assert summarizer.done


@pytest.mark.asyncio
async def test_summary_factory_error_aborts_loop() -> None:
    """A factory that raises journals + returns; the loop doesn't loop forever."""
    store = TasksStore()
    rec = store.create(description="d", prompt="p")

    def _broken() -> Any:
        raise RuntimeError("no summary model for you")

    # Loop should exit gracefully; not raise.
    await asyncio.wait_for(
        run_summary_loop(
            task_id=rec.id,
            store=store,
            transcript_provider=lambda: [HumanMessage(content="hi")],
            summary_model_factory=_broken,
            interval_sec=0.01,
        ),
        timeout=1.0,
    )
    refreshed = store.get(rec.id)
    assert refreshed is not None
    # Factory failed before any summary landed.
    assert refreshed.progress.latest_summary is None


# ---------------------------------------------------------------------------
# Summary model factory (llm.make_summary_model_factory)
# ---------------------------------------------------------------------------


def test_summary_factory_falls_back_to_main_model_when_no_spec() -> None:
    cfg = _cfg()
    main = FakeChatModel(turns=[])
    factory = llm.make_summary_model_factory(cfg, main, summary_spec=None)
    assert factory() is main
    # Memoization: repeated calls return the same instance.
    assert factory() is main


def test_summary_factory_uses_summary_spec_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summary_spec pointing at a real provider routes through resolve+create."""
    cfg = AuraConfig.model_validate({
        "providers": [
            {"name": "openai", "protocol": "openai", "api_key": "test-key"},
        ],
        "router": {
            "default": "openai:gpt-4o-mini",
            "cheap": "openai:gpt-4o-mini",
        },
        "tools": {"enabled": []},
    })
    main = FakeChatModel(turns=[])
    sentinel: list[FakeChatModel] = []

    def _create_stub(provider: Any, model_name: str) -> FakeChatModel:
        m = FakeChatModel(turns=[])
        sentinel.append(m)
        return m

    monkeypatch.setattr(llm, "create", _create_stub)
    factory = llm.make_summary_model_factory(cfg, main, summary_spec="cheap")
    out1 = factory()
    out2 = factory()
    # Memoized: only one create call.
    assert out1 is out2
    assert out1 is sentinel[0]
    assert len(sentinel) == 1
    # NOT the main model — the cheap one.
    assert out1 is not main


# ---------------------------------------------------------------------------
# End-to-end via run_task
# ---------------------------------------------------------------------------


def _spawn_factory(
    *, sub_response: AIMessage,
) -> tuple[TasksStore, SubagentFactory, Callable[[], FakeChatModel]]:
    store = TasksStore()
    main_model = FakeChatModel(turns=[FakeTurn(sub_response)])

    def _factory_callable() -> FakeChatModel:
        return main_model

    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=_factory_callable,
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    return store, factory, _factory_callable


@pytest.mark.asyncio
async def test_summary_pulled_into_terminal_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: spawn a slow subagent, summary lands on the record before
    the terminal mark.

    We patch ``make_summary_model_factory`` to hand back a deterministic
    summary model so the test doesn't depend on any external SDK.
    """
    summary_model, _captured = _make_summary_model([
        "cataloging the project files",
        "reading config",
    ])

    monkeypatch.setattr(
        llm,
        "make_summary_model_factory",
        lambda cfg, main, *, summary_spec=None: (lambda: summary_model),
    )

    # The subagent's main model must take long enough for at least one
    # summarizer tick to fire. Wrap _agenerate with a 0.1s sleep.
    sub_response = AIMessage(
        content="all done",
        usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        },
    )
    store, factory, _ = _spawn_factory(sub_response=sub_response)

    # Wrap the main model's _agenerate to add a delay so the
    # summarizer can fire. ``_model_factory`` is a private attr but
    # tests legitimately reach in to swap behavior.
    factory_mf: Any = factory._model_factory  # noqa: SLF001
    sub_model = factory_mf()
    orig = sub_model._agenerate

    async def _slow(*a: Any, **kw: Any) -> Any:
        await asyncio.sleep(0.08)
        return await orig(*a, **kw)
    sub_model._agenerate = _slow
    # The factory builds a fresh model on each spawn — re-bind so spawn
    # picks up the slow version.
    factory._model_factory = lambda: sub_model  # noqa: SLF001

    rec = store.create(description="probe", prompt="go")
    await run_task(store, factory, rec.id, summary_interval_sec=0.03)

    refreshed = store.get(rec.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    # At least one summary landed before terminal.
    assert refreshed.progress.latest_summary is not None
    # Token count also landed.
    assert refreshed.progress.token_count == 15
