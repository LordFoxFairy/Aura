"""Subagent ``.meta.json`` companion writer (claude-code parity).

Each persisted subagent transcript JSONL is shadowed by a sibling
``.meta.json`` file carrying the small header an operator wants when
scanning a session directory: agent_type, started_at, status, token
totals — without having to load and parse the full transcript.

Track A is delivering the on-disk path API
(``SessionStorage.subagent_metadata_path(task_id, *, parent_session_id,
cwd)``). Until A lands, the writer here defensively no-ops with a
journal warning. These tests stub the API in via monkeypatch so they
exercise the writer end-to-end regardless of A's landing order.

Pinned behaviors:

- A ``.meta.json`` file lands on every terminal branch (completed,
  failed, cancelled, timeout).
- The schema matches the documented contract: agent_type, task_id,
  description, model_spec, parent_session_id, cwd, started_at,
  ended_at, status, input_tokens, output_tokens.
- Status echoes ``TaskRecord.status`` exactly.
- Token totals reflect what the post_model hook recorded.
- The write is atomic via a ``.tmp`` rename — a torn write can never
  leave half-serialized JSON visible.
- A storage that throws from ``subagent_metadata_path`` does NOT block
  the agent loop; the subagent still flips to its terminal status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from aura.config.schema import AuraConfig
from aura.core.persistence.storage import SessionStorage
from aura.core.tasks.factory import SubagentFactory
from aura.core.tasks.run import run_task
from aura.core.tasks.store import TasksStore
from tests.conftest import FakeChatModel, FakeTurn

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_factory(
    *,
    turns: list[FakeTurn] | None = None,
) -> SubagentFactory:
    """Build a SubagentFactory with a one-turn FakeChatModel by default."""
    return SubagentFactory(
        parent_config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(
            turns=turns or [FakeTurn(AIMessage(content="child-final"))],
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )


def _meta_path_for(
    storage: SessionStorage,
    task_id: str,
    *,
    parent_session_id: str | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve the on-disk meta path via Track A's storage API."""
    return storage.subagent_metadata_path(
        task_id,
        parent_session_id=parent_session_id,
        cwd=cwd,
    )


def _read_meta(
    storage: SessionStorage,
    task_id: str,
    *,
    parent_session_id: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    path = _meta_path_for(
        storage,
        task_id,
        parent_session_id=parent_session_id,
        cwd=cwd,
    )
    assert path.exists(), f"meta file not found at {path}"
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return raw


# ----------------------------------------------------------------------
# Tests — terminal branches
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_meta_json_written_on_terminal_completion(
    tmp_path: Path,
) -> None:
    """Happy path: subagent completes -> meta file lands on disk."""
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-completed", prompt="hi")

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="user-session-A",
        cwd=str(tmp_path),
    )

    meta = _read_meta(
        storage, rec.id,
        parent_session_id="user-session-A",
        cwd=tmp_path,
    )
    assert meta["task_id"] == rec.id
    assert meta["status"] == "completed"
    assert meta["description"] == "d-completed"
    assert meta["agent_type"] == "general-purpose"
    assert meta["parent_session_id"] == "user-session-A"
    assert meta["started_at"] > 0
    assert meta["ended_at"] is not None
    assert meta["ended_at"] >= meta["started_at"]


@pytest.mark.asyncio
async def test_meta_json_written_on_terminal_failure(
    tmp_path: Path,
) -> None:
    """Failure path: child raises mid-stream -> meta lands with failed."""
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()

    class _MidStreamBoomAgent:
        """Agent stub whose astream raises after first event."""

        _hooks = None
        _config = None
        _model = None
        _storage = None
        _session_id = None

        async def astream(self, _prompt: str) -> Any:  # noqa: ANN401
            raise RuntimeError("boom-mid-stream")
            yield  # pragma: no cover — make this a generator

        async def aclose(self) -> None:
            return None

    class _StubFactory(SubagentFactory):
        def spawn(self, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            return _MidStreamBoomAgent()

    factory = _StubFactory(
        parent_config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    rec = store.create(description="d-failed", prompt="hi")

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="parent-fail",
        cwd=str(tmp_path),
    )

    rec_after = store.get(rec.id)
    assert rec_after is not None
    assert rec_after.status == "failed"
    meta = _read_meta(
        storage, rec.id,
        parent_session_id="parent-fail",
        cwd=tmp_path,
    )
    assert meta["status"] == "failed"
    assert meta["task_id"] == rec.id


@pytest.mark.asyncio
async def test_meta_json_written_on_terminal_cancelled(
    tmp_path: Path,
) -> None:
    """Cancel path: child gets cancelled -> meta lands with cancelled."""
    import asyncio as _asyncio

    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()

    # Build an agent stub whose astream awaits forever — that gives a
    # deterministic window for the outer cancel to land *inside* the
    # asyncio.timeout block in run_task, exercising the
    # CancelledError branch (and not the success / failure branches).
    class _ForeverAgent:
        _hooks = None
        _config = None
        _model = None
        _storage = None
        _session_id = None

        async def astream(self, _prompt: str) -> Any:  # noqa: ANN401
            await _asyncio.Event().wait()
            yield  # pragma: no cover

        async def aclose(self) -> None:
            return None

    class _ForeverFactory(SubagentFactory):
        def spawn(self, *_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            return _ForeverAgent()

    factory = _ForeverFactory(
        parent_config=AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": []},
        }),
        parent_model_spec="openai:gpt-4o-mini",
        model_factory=lambda: FakeChatModel(),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    rec = store.create(description="d-cancelled", prompt="hi")

    async def _runner() -> None:
        await run_task(
            store, factory, rec.id,
            transcript_storage=storage,
            parent_session_id="parent-cancel",
            cwd=str(tmp_path),
        )

    handle = _asyncio.create_task(_runner())
    # Let run_task get past spawn + into the astream wait.
    for _ in range(5):
        await _asyncio.sleep(0)
    handle.cancel()
    with pytest.raises(_asyncio.CancelledError):
        await handle

    rec_after = store.get(rec.id)
    assert rec_after is not None
    assert rec_after.status == "cancelled"
    meta = _read_meta(
        storage, rec.id,
        parent_session_id="parent-cancel",
        cwd=tmp_path,
    )
    assert meta["status"] == "cancelled"
    assert meta["task_id"] == rec.id


@pytest.mark.asyncio
async def test_meta_json_includes_input_output_tokens(
    tmp_path: Path,
) -> None:
    """Token totals from the post_model hook flow into the meta file."""
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-tokens", prompt="hi")
    # Inject tokens directly via the store API — equivalent to what
    # the post_model observer would do at runtime, but deterministic.
    store.record_token_usage(rec.id, input_tokens=42, output_tokens=99)

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="parent-tokens",
        cwd=str(tmp_path),
        # Disable summarizer so the test doesn't need the LLM factory.
        summary_interval_sec=0,
    )

    meta = _read_meta(
        storage, rec.id,
        parent_session_id="parent-tokens",
        cwd=tmp_path,
    )
    assert meta["input_tokens"] == 42
    assert meta["output_tokens"] == 99


@pytest.mark.asyncio
async def test_meta_json_atomic_write_via_tmp(
    tmp_path: Path,
) -> None:
    """The writer must use a ``.tmp`` sibling + rename for atomicity.

    We verify by checking no orphan ``.tmp`` survives a successful
    run and the final file exists. The atomic-rename pattern
    guarantees a torn write can never leave half-serialized JSON
    visible at the canonical path.
    """
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-atomic", prompt="hi")

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="parent-atomic",
        cwd=str(tmp_path),
    )

    final = _meta_path_for(
        storage, rec.id,
        parent_session_id="parent-atomic",
        cwd=tmp_path,
    )
    assert final.exists()
    # No orphan tmp left behind in the meta's parent directory.
    leftover_tmps = list(final.parent.glob("*.tmp"))
    assert leftover_tmps == [], (
        f"orphan .tmp files: {leftover_tmps}"
    )


@pytest.mark.asyncio
async def test_meta_write_failure_does_not_block_loop(
    tmp_path: Path,
) -> None:
    """Storage path API raising must NOT propagate into run_task."""
    storage = SessionStorage(tmp_path / "parent.db")

    def _raising_path(
        _task_id: str,
        *,
        parent_session_id: str | None = None,  # noqa: ARG001
        cwd: Path | None = None,  # noqa: ARG001
    ) -> Path:
        raise OSError("disk full")

    # Shadow the bound method on the instance with a function that raises.
    storage.subagent_metadata_path = _raising_path  # type: ignore[assignment]

    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-resilient", prompt="hi")

    # MUST NOT raise.
    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
    )

    rec_after = store.get(rec.id)
    assert rec_after is not None
    # The subagent's terminal mark fires regardless of meta failure.
    assert rec_after.status == "completed"


@pytest.mark.asyncio
async def test_meta_status_field_matches_record_status(
    tmp_path: Path,
) -> None:
    """The meta's ``status`` field is verbatim from TaskRecord.status."""
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-status", prompt="hi")

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="parent-status",
        cwd=str(tmp_path),
    )

    rec_after = store.get(rec.id)
    assert rec_after is not None
    meta = _read_meta(
        storage, rec.id,
        parent_session_id="parent-status",
        cwd=tmp_path,
    )
    assert meta["status"] == rec_after.status


@pytest.mark.asyncio
async def test_meta_skipped_when_storage_lacks_path_api(
    tmp_path: Path,
) -> None:
    """Defensive getattr path: storage missing the API → silent skip.

    Track A has now landed ``subagent_metadata_path`` on
    :class:`SessionStorage`, but the writer remains defensive (read
    via ``getattr``) so a future stripped-down storage subclass — or
    a regression that drops the method — still produces a working
    subagent run with only a journal warning. We exercise that path
    by shadowing the bound method on the instance with ``None``.
    """
    storage = SessionStorage(tmp_path / "parent.db")
    # Shadow the class method on the instance. ``getattr(storage,
    # "subagent_metadata_path", None)`` returns this ``None`` rather
    # than the real method, so the writer hits its skip branch.
    storage.subagent_metadata_path = None  # type: ignore[assignment]

    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-no-api", prompt="hi")

    # MUST NOT raise.
    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
    )

    rec_after = store.get(rec.id)
    assert rec_after is not None
    assert rec_after.status == "completed"
    # No meta file landed — but transcript still did.
    transcript_root = tmp_path / "subagents"
    if transcript_root.exists():
        meta_files = list(transcript_root.glob("*.meta.json"))
        assert meta_files == []


@pytest.mark.asyncio
async def test_meta_iso_timestamps_present(
    tmp_path: Path,
) -> None:
    """ISO-8601 sibling fields exist for human-friendly inspection."""
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(description="d-iso", prompt="hi")

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="parent-iso",
        cwd=str(tmp_path),
    )

    meta = _read_meta(
        storage, rec.id,
        parent_session_id="parent-iso",
        cwd=tmp_path,
    )
    assert isinstance(meta["started_at_iso"], str)
    assert "T" in meta["started_at_iso"]  # ISO-8601 marker
    assert meta["ended_at_iso"] is not None
    assert "T" in meta["ended_at_iso"]


@pytest.mark.asyncio
async def test_meta_includes_full_schema_fields(
    tmp_path: Path,
) -> None:
    """Pin the documented schema — every key must be present."""
    storage = SessionStorage(tmp_path / "parent.db")
    store = TasksStore()
    factory = _make_factory()
    rec = store.create(
        description="d-schema",
        prompt="hi",
        agent_type="general-purpose",
        model_spec="openai:gpt-4o-mini",
    )

    await run_task(
        store, factory, rec.id,
        transcript_storage=storage,
        parent_session_id="parent-XYZ",
        cwd=str(tmp_path),
    )

    meta = _read_meta(
        storage, rec.id,
        parent_session_id="parent-XYZ",
        cwd=tmp_path,
    )
    expected_keys = {
        "agent_type",
        "task_id",
        "description",
        "model_spec",
        "parent_session_id",
        "cwd",
        "started_at",
        "started_at_iso",
        "ended_at",
        "ended_at_iso",
        "status",
        "input_tokens",
        "output_tokens",
    }
    missing = expected_keys - set(meta.keys())
    assert missing == set(), f"meta missing keys: {missing}"
    assert meta["model_spec"] == "openai:gpt-4o-mini"
    assert meta["parent_session_id"] == "parent-XYZ"
    assert meta["cwd"] == str(tmp_path)


# Anchor unused-import for HumanMessage (kept for future expansion of
# transcript-aware tests; Ruff won't flag unused after first reference).
_ = HumanMessage
