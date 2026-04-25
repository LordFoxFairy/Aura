"""Per-spawn model override (axis 18 — claude-code parity).

These tests pin the contract that ``task_create(model="...")`` overrides
the parent's inherited model for THIS subagent only, and that
``model=None`` (the default) inherits the parent. The TaskRecord
remembers the resolved spec so ``task_get`` / ``task_list`` can show
which model the child actually ran on.

Strategy: stub ``llm._load_class`` so ``make_model_for_spec`` returns a
recognisable shim instead of hitting a real SDK. The tests assert the
spec string was routed correctly + the TaskRecord persists it. The
existing model_factory injection path (FakeChatModel) is exercised
elsewhere; here we deliberately use the resolve path so we can observe
which spec was passed in.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from aura.config.schema import AuraConfig
from aura.core import llm
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.store import TasksStore
from aura.schemas.tool import ToolError
from aura.tools.task_create import TaskCreate, TaskCreateParams
from aura.tools.task_get import TaskGet


def _cfg() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [
            {"name": "openai", "protocol": "openai"},
            {"name": "anthropic", "protocol": "anthropic"},
        ],
        "router": {
            "default": "openai:gpt-4o-mini",
            "haiku": "anthropic:claude-haiku-4-5",
        },
        "tools": {"enabled": []},
    })


class _StubModel:
    """Minimal stand-in for a chat model — just remembers its kwargs.

    We don't need ``BaseChatModel`` semantics; the factory hands the
    model through to Agent's constructor, and the per-spawn-model tests
    assert on which spec resolved at the make_model_for_spec layer (so
    the model object never actually drives a turn).
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_load_class(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``llm._load_class`` so model construction records its kwargs.

    Returns the live list the stub appends to — tests assert against the
    captured kwargs (notably ``model=...``) to confirm the right spec
    flowed through.
    """
    captured: list[dict[str, Any]] = []

    class _RecordingStub(_StubModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured.append(kwargs)

    monkeypatch.setattr(llm, "_load_class", lambda _proto: _RecordingStub)
    # Both providers expect API keys via env; populate so credential
    # resolution doesn't trip the test.
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    return captured


def test_make_model_for_spec_routes_through_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_load_class(monkeypatch)
    cfg = _cfg()
    model = llm.make_model_for_spec("anthropic:claude-haiku-4-5", cfg)
    assert isinstance(model, _StubModel)
    assert captured[-1]["model"] == "claude-haiku-4-5"


def test_make_model_for_spec_resolves_router_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_load_class(monkeypatch)
    cfg = _cfg()
    llm.make_model_for_spec("haiku", cfg)
    # Router maps "haiku" -> "anthropic:claude-haiku-4-5"; the stub sees
    # the right-hand-side model_name.
    assert captured[-1]["model"] == "claude-haiku-4-5"


def test_make_model_for_spec_invalid_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load_class(monkeypatch)
    cfg = _cfg()
    with pytest.raises(llm.UnknownModelSpecError):
        llm.make_model_for_spec("ghost-provider:m", cfg)


@pytest.mark.asyncio
async def test_default_inherits_parent_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No override → factory uses the parent's model_spec."""
    captured = _patch_load_class(monkeypatch)
    cfg = _cfg()
    store = TasksStore()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
    out = await tool.ainvoke({"description": "d", "prompt": "p"})
    # Drain the detached task so it actually runs (and resolves a model).
    task_id = out["task_id"]
    handle = tasks.get(task_id)
    if handle is not None:
        await asyncio.wait_for(handle, timeout=2.0)
    # The TaskRecord remembers the parent's spec when no override given.
    rec = store.get(task_id)
    assert rec is not None
    assert rec.model_spec == "openai:gpt-4o-mini"
    # And the model that was actually built used the parent's model_name.
    assert any(c.get("model") == "gpt-4o-mini" for c in captured), (
        f"expected parent's gpt-4o-mini in resolved kwargs, got {captured!r}"
    )


@pytest.mark.asyncio
async def test_override_uses_specified_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured = _patch_load_class(monkeypatch)
    cfg = _cfg()
    store = TasksStore()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
    out = await tool.ainvoke({
        "description": "cheap-explore",
        "prompt": "find TODOs",
        "model": "anthropic:claude-haiku-4-5",
    })
    task_id = out["task_id"]
    handle = tasks.get(task_id)
    if handle is not None:
        await asyncio.wait_for(handle, timeout=2.0)
    # Persisted spec on the record reflects the override.
    rec = store.get(task_id)
    assert rec is not None
    assert rec.model_spec == "anthropic:claude-haiku-4-5"
    assert out["model_spec"] == "anthropic:claude-haiku-4-5"
    # Resolution went through the override, not the parent's gpt-4o-mini.
    assert any(c.get("model") == "claude-haiku-4-5" for c in captured), (
        f"expected override claude-haiku-4-5 in kwargs, got {captured!r}"
    )


@pytest.mark.asyncio
async def test_invalid_spec_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_load_class(monkeypatch)
    cfg = _cfg()
    store = TasksStore()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
    with pytest.raises(ToolError) as ei:
        await tool.ainvoke({
            "description": "d",
            "prompt": "p",
            "model": "ghost-provider:nope",
        })
    msg = str(ei.value)
    assert "ghost-provider" in msg or "ghost" in msg
    assert "invalid model spec" in msg
    # No orphan record left behind by the failed validation.
    assert store.list() == []


def test_run_in_background_field_accepted() -> None:
    """The run_in_background field must round-trip through the schema."""
    params = TaskCreateParams.model_validate({
        "description": "d",
        "prompt": "p",
        "run_in_background": True,
    })
    assert params.run_in_background is True
    # Default is False.
    default_params = TaskCreateParams.model_validate({
        "description": "d",
        "prompt": "p",
    })
    assert default_params.run_in_background is False


@pytest.mark.asyncio
async def test_task_record_remembers_model_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Even without override, the resolved spec is pinned on the record."""
    _patch_load_class(monkeypatch)
    cfg = _cfg()
    store = TasksStore()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tool = TaskCreate(store=store, factory=factory, running=tasks)
    out = await tool.ainvoke({"description": "d", "prompt": "p"})
    rec = store.get(out["task_id"])
    assert rec is not None
    # Set IMMEDIATELY at create time — not lazily on first activity.
    assert rec.model_spec == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_task_get_returns_model_spec_in_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _patch_load_class(monkeypatch)
    cfg = _cfg()
    store = TasksStore()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    tc = TaskCreate(store=store, factory=factory, running=tasks)
    out = await tc.ainvoke({
        "description": "d",
        "prompt": "p",
        "model": "haiku",  # router alias path
    })
    tg = TaskGet(store=store)
    info = await tg.ainvoke({"task_id": out["task_id"]})
    assert info["model_spec"] == "haiku"
    # agent_type also surfaced (regression guard for the new task_get key).
    assert info["agent_type"] == "general-purpose"


def test_factory_validate_model_spec_passes_for_router_alias() -> None:
    cfg = _cfg()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    # Should not raise — "haiku" is a router alias defined in _cfg().
    factory.validate_model_spec("haiku")
    factory.validate_model_spec("anthropic:claude-haiku-4-5")


def test_factory_validate_model_spec_rejects_unknown_provider() -> None:
    cfg = _cfg()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    with pytest.raises(llm.UnknownModelSpecError):
        factory.validate_model_spec("ghost-provider:nope")


def test_factory_parent_model_spec_property() -> None:
    cfg = _cfg()
    factory = SubagentFactory(
        parent_config=cfg,
        parent_model_spec="openai:gpt-4o-mini",
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    assert factory.parent_model_spec == "openai:gpt-4o-mini"
