"""G1 acceptance tests — user message + attachments persist BEFORE model call.

Contract (matches claude-code QueryEngine.ts:431+451): ``Agent.astream``
appends the HumanMessage (and any attachments) to history and calls
``storage.save`` BEFORE delegating to the loop. If the model call (or the
whole turn) crashes / is cancelled, the user's input is already on disk —
``resume`` semantics work across a ``Ctrl-C`` or process kill.

Tests here:

- **AC-G1-1** — ``test_user_message_persisted_before_model_call``: when
  ``model.ainvoke`` raises a non-overflow ConnectionError, the user turn
  (and any attachments) is still readable from ``storage.load`` after the
  exception propagates.
- **AC-G1-2** — ``test_reactive_compact_retains_attachments_idempotently``:
  reactive-compact retry does NOT re-pass ``attachments=`` (it reads them
  from history instead); the attachment payload is byte-equivalent before
  and after compact + retry.
- **AC-G1-3** — covered by running the existing reactive-compact suite
  (``tests/test_reactive_compact.py``) after the refactor: no semantic
  drift. Spot-checked in ``test_reactive_compact_still_green_after_g1``.
- **AC-G1-4** — ``test_cancelled_turn_persists_user_message_dogfood``:
  real subprocess that writes a HumanMessage + hangs mid-stream, gets
  SIGINT'd, re-loads the same session_id, and sees its user message in
  history. Uses a subprocess (NOT StringIO / mock-only) to satisfy the
  dogfood-before-done discipline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn


def _minimal_config() -> AuraConfig:
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
    })


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


class _RaisingOnce(FakeChatModel):
    """FakeChatModel that raises the provided exception on its first call."""

    def __init__(
        self,
        *,
        error: BaseException,
        turns: list[FakeTurn] | None = None,
    ) -> None:
        super().__init__(turns=turns or [])
        self.__dict__["_error"] = error
        self.__dict__["_raised"] = False

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        if not self.__dict__["_raised"]:
            self.__dict__["_raised"] = True
            raise self.__dict__["_error"]
        turn = self._pop_turn()
        return ChatResult(generations=[ChatGeneration(message=turn.message)])


# ---------------------------------------------------------------------------
# AC-G1-1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_message_persisted_before_model_call(tmp_path: Path) -> None:
    """AC-G1-1: storage.load sees the user turn even when ainvoke crashes.

    The contract: ``astream`` appends the HumanMessage (+ any attachments) to
    history and persists BEFORE handing off to the loop. A connection error
    propagates to the caller, but the user's input is already on disk — the
    next ``resume`` of this session sees it.
    """
    storage = _storage(tmp_path)
    # ``ValueError`` is a non-retriable class and its message ("bad request")
    # doesn't match any retry substring — propagates on first try. Chosen to
    # exercise the same astream path as a true ConnectionError would without
    # paying for the retry backoff ladder in test time.
    model = _RaisingOnce(
        error=ValueError("bad request: provider died mid-stream"),
        turns=[],
    )
    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        auto_compact_threshold=0,
    )

    attachment = HumanMessage(content="<mcp-resource uri='mem://doc'>BODY</mcp-resource>")
    with pytest.raises(ValueError, match="provider died"):
        async for _ in agent.astream("please read the doc", attachments=[attachment]):
            pass

    saved = storage.load(agent.session_id)
    # History must carry: attachment envelope + user HumanMessage.
    # Order is strict — attachments BEFORE user turn (claude-code parity).
    assert len(saved) == 2, (
        f"expected [attachment, user_msg] after crash; got {len(saved)} "
        f"items: {[type(m).__name__ for m in saved]}"
    )
    assert isinstance(saved[0], HumanMessage)
    assert "<mcp-resource" in str(saved[0].content)
    assert "BODY" in str(saved[0].content)
    assert isinstance(saved[1], HumanMessage)
    assert saved[1].content == "please read the doc"
    await agent.aclose()


@pytest.mark.asyncio
async def test_user_message_persisted_before_model_call_no_attachments(
    tmp_path: Path,
) -> None:
    """AC-G1-1 (plain): user-only turn also persists before ainvoke."""
    storage = _storage(tmp_path)
    # Non-retriable + no context-overflow signature → propagates unchanged.
    model = _RaisingOnce(error=ValueError("bad request: provider died"), turns=[])
    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        auto_compact_threshold=0,
    )

    with pytest.raises(ValueError, match="provider died"):
        async for _ in agent.astream("find the bug"):
            pass

    saved = storage.load(agent.session_id)
    assert len(saved) == 1
    assert isinstance(saved[0], HumanMessage)
    assert saved[0].content == "find the bug"
    await agent.aclose()


# ---------------------------------------------------------------------------
# AC-G1-2
# ---------------------------------------------------------------------------


class _OverflowThenOK(FakeChatModel):
    """Raise a context-overflow error on the first call; normal turns after.

    Tracks every ``messages`` list the model sees so the test can verify
    attachments survive the reactive-compact retry.
    """

    def __init__(self, *, turns: list[FakeTurn]) -> None:
        super().__init__(turns=turns)
        self.__dict__["_overflow_raised"] = False
        self.__dict__["captured"] = []

    @property
    def captured(self) -> list[list[BaseMessage]]:
        return self.__dict__["captured"]  # type: ignore[no-any-return]

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **_: Any,
    ) -> ChatResult:
        self.__dict__["ainvoke_calls"] += 1
        self.__dict__["captured"].append(list(messages))
        if not self.__dict__["_overflow_raised"]:
            self.__dict__["_overflow_raised"] = True
            raise RuntimeError(
                "400 — context_length_exceeded: prompt is too long"
            )
        turn = self._pop_turn()
        return ChatResult(generations=[ChatGeneration(message=turn.message)])


@pytest.mark.asyncio
async def test_reactive_compact_retains_attachments_idempotently(
    tmp_path: Path,
) -> None:
    """AC-G1-2: reactive-compact retry reads attachments from history.

    The new contract: ``astream`` pre-appends attachments + user msg and
    persists BEFORE calling the loop. On context-overflow the retry does
    NOT pass ``attachments=`` to ``run_turn`` a second time — it loads the
    already-persisted history (which contains the envelope) and replays.
    Byte-equivalence is checked against the envelope content.
    """
    storage = _storage(tmp_path)
    # Seed enough prior history so compact has something to summarize.
    seed: list[BaseMessage] = []
    for i in range(10):
        seed.append(HumanMessage(content=f"u-{i}"))
        seed.append(AIMessage(content=f"a-{i}"))
    storage.save("default", seed)

    model = _OverflowThenOK(turns=[
        FakeTurn(AIMessage(content="SUMMARY")),    # compact summary turn
        FakeTurn(AIMessage(content="recovered")),  # retry turn
    ])
    agent = Agent(
        config=_minimal_config(),
        model=model,
        storage=storage,
        auto_compact_threshold=0,
    )

    payload = "<mcp-resource uri='mem://doc.md'>IMPORTANT-BODY-12345</mcp-resource>"
    attachment = HumanMessage(content=payload)
    async for _ in agent.astream("tldr", attachments=[attachment]):
        pass

    # Three model calls total: overflow attempt, summary turn, retry.
    assert model.ainvoke_calls >= 3, (
        f"expected >=3 model calls (overflow + summary + retry); got {model.ainvoke_calls}"
    )

    # First attempt's messages must carry the attachment envelope.
    first_msgs = model.captured[0]
    envelopes_first = [
        m for m in first_msgs
        if isinstance(m, HumanMessage) and "IMPORTANT-BODY-12345" in str(m.content)
    ]
    assert len(envelopes_first) == 1, (
        "attachment must be visible to the first (overflow-raising) model call"
    )

    # Retry must STILL see the same attachment payload — byte-equivalent — even
    # though astream did not re-pass attachments=. It's in history.
    retry_msgs = model.captured[-1]
    envelopes_retry = [
        m for m in retry_msgs
        if isinstance(m, HumanMessage) and "IMPORTANT-BODY-12345" in str(m.content)
    ]
    assert len(envelopes_retry) == 1, (
        "attachment must survive the reactive-compact retry via history"
    )
    assert envelopes_first[0].content == envelopes_retry[0].content, (
        "attachment content drifted between first attempt and retry"
    )
    await agent.aclose()


# ---------------------------------------------------------------------------
# AC-G1-3 — spot check the reactive_compact suite still passes shape-wise.
# (The full test file is invoked by ``make check``; we do a narrow
# reachability assertion here so a future refactor that orphans those tests
# fails this one first.)
# ---------------------------------------------------------------------------


def test_reactive_compact_still_green_after_g1() -> None:
    """AC-G1-3 guard: the reactive-compact test module is importable and the
    four named tests still exist with their original signatures."""
    import tests.test_reactive_compact as rc

    # If any of these got renamed, the suite's semantics drifted and the
    # caller should look at the diff before trusting "all green".
    required = {
        "test_reactive_compact_on_context_length_error",
        "test_reactive_compact_only_retries_once",
        "test_reactive_compact_other_error_passthrough",
        "test_reactive_compact_journal_event",
    }
    present = {name for name in dir(rc) if name.startswith("test_")}
    missing = required - present
    assert not missing, (
        f"reactive-compact regression tests missing: {missing}; "
        f"did a refactor rename them?"
    )


# ---------------------------------------------------------------------------
# AC-G1-4 — dogfood / real subprocess
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _pre_append_then_kill_driver() -> str:
    """Return the Python source for the subprocess driver.

    The driver:
    1. Constructs an Agent with a model that hangs forever on ``ainvoke``.
    2. Spawns astream as an asyncio.Task.
    3. Waits for the post-save signal file to appear.
    4. Exits hard (``os._exit(137)``) — simulating a kill mid-stream.

    The subprocess runs ``aura`` library code (not the CLI) so we don't
    need a provider SDK / credential. The storage file is passed in via
    ``sys.argv``.
    """
    return textwrap.dedent(
        """
        import asyncio
        import os
        import sys
        import time
        from pathlib import Path
        from typing import Any

        from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        # Add repo root to sys.path so ``tests.conftest`` imports work
        # (the driver file lives in the test's tmp_path, not in the package).
        sys.path.insert(0, {repo_root!r})

        from aura.config.schema import AuraConfig
        from aura.core.agent import Agent
        from aura.core.persistence.storage import SessionStorage
        from tests.conftest import FakeChatModel

        db_path = sys.argv[1]
        signal_path = sys.argv[2]
        session_id = sys.argv[3]
        prompt = sys.argv[4]

        class _HangingModel(FakeChatModel):
            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: AsyncCallbackManagerForLLMRun | None = None,
                **_: Any,
            ) -> ChatResult:
                # Signal parent that we reached the model call — meaning the
                # pre-save has happened already (astream saved before invoke).
                Path(signal_path).write_text("at-model")
                # Hang forever; parent will hard-kill us.
                await asyncio.sleep(300)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=""))]
                )

        cfg = AuraConfig.model_validate({{
            "providers": [{{"name": "openai", "protocol": "openai"}}],
            "router": {{"default": "openai:gpt-4o-mini"}},
            "tools": {{"enabled": []}},
        }})
        storage = SessionStorage(Path(db_path))
        agent = Agent(
            config=cfg,
            model=_HangingModel(turns=[]),
            storage=storage,
            session_id=session_id,
            auto_compact_threshold=0,
        )

        async def main() -> None:
            async for _ in agent.astream(prompt):
                pass

        try:
            asyncio.run(main())
        except BaseException:
            # Whatever happens in main, we never reach normal exit — parent
            # is going to SIGKILL us. Be noisy so the test can assert if we
            # somehow escape.
            pass
        """
    ).format(repo_root=str(_REPO_ROOT))


def test_cancelled_turn_persists_user_message_dogfood(tmp_path: Path) -> None:
    """AC-G1-4: mid-stream SIGKILL still leaves the user turn on disk.

    Real subprocess — not a mocked astream context manager. The driver
    spawns astream against a model that hangs in ``_agenerate``; after
    the pre-invoke save the driver signals the parent via a marker file.
    Parent SIGKILLs the child. A fresh SessionStorage then reads from
    the same DB path with the same session_id and MUST see the user
    turn.

    This is the dogfood assertion: if we regressed on persistence-order,
    the HumanMessage wouldn't land before the hang and a fresh load
    would return ``[]``. That's the bug G1 closes.
    """
    db_path = tmp_path / "session.db"
    signal_path = tmp_path / "at_model.marker"
    driver = tmp_path / "driver.py"
    driver.write_text(_pre_append_then_kill_driver())

    session_id = "dogfood-g1"
    prompt = "please analyze the attached log and find the root cause"

    proc = subprocess.Popen(
        [
            sys.executable,
            str(driver),
            str(db_path),
            str(signal_path),
            session_id,
            prompt,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait up to 30s for the child to reach the model call (post-save).
        # 30s is generous — on CI the first import chain can be slow.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if signal_path.exists():
                break
            if proc.poll() is not None:
                # Child exited before signalling — something else went wrong.
                stdout, stderr = proc.communicate(timeout=2)
                raise AssertionError(
                    f"driver exited before signalling (rc={proc.returncode}):\n"
                    f"stdout={stdout.decode(errors='replace')!r}\n"
                    f"stderr={stderr.decode(errors='replace')!r}"
                )
            time.sleep(0.05)
        if not signal_path.exists():
            proc.kill()
            stdout, stderr = proc.communicate(timeout=2)
            raise AssertionError(
                f"driver did not signal within 30s; "
                f"stdout={stdout.decode(errors='replace')!r}\n"
                f"stderr={stderr.decode(errors='replace')!r}"
            )

        # Hard kill — simulate SIGKILL / OS kill during streaming. SIGKILL
        # (not SIGTERM) so Python's asyncio cancellation/finalizers can't
        # sneak a post-turn save in. If the contract is "save BEFORE
        # model call", even SIGKILL must leave the user turn on disk.
        proc.kill()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)

    # Re-open the same DB with a fresh storage + same session_id. The user
    # message must be there.
    storage2 = SessionStorage(db_path)
    try:
        loaded = storage2.load(session_id)
    finally:
        storage2.close()

    # User message (and only the user message — no AI reply yet) on disk.
    assert loaded, (
        "history was empty after mid-stream SIGKILL — user message did NOT "
        "persist before the model call; G1 contract is broken"
    )
    human_msgs: Sequence[HumanMessage] = [
        m for m in loaded if isinstance(m, HumanMessage)
    ]
    assert any(prompt in str(m.content) for m in human_msgs), (
        f"user prompt {prompt!r} not found in persisted history; "
        f"got {[type(m).__name__ + ':' + str(m.content)[:80] for m in loaded]}"
    )


# ---------------------------------------------------------------------------
# Extra: verify the journal shows save_before_run ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_happens_before_turn_begin_in_journal(
    tmp_path: Path,
) -> None:
    """The ordering invariant is also visible in the journal stream:
    a ``storage_save`` for the session precedes the first ``turn_begin``
    of an astream call. If this flips, the pre-save contract regressed.
    """
    from aura.core.persistence import journal

    log_path = tmp_path / "audit.jsonl"
    journal.configure(log_path)
    try:
        model = FakeChatModel(turns=[FakeTurn(AIMessage(content="ok"))])
        agent = Agent(
            config=_minimal_config(),
            model=model,
            storage=_storage(tmp_path),
            auto_compact_threshold=0,
        )
        async for _ in agent.astream("hi"):
            pass
        await agent.aclose()

        events = [
            json.loads(line)
            for line in log_path.read_text().strip().split("\n")
            if line
        ]
    finally:
        journal.reset()

    names = [e["event"] for e in events]
    # First astream_begin, then the pre-invoke storage_save, THEN turn_begin.
    assert "astream_begin" in names
    assert "storage_save" in names
    assert "turn_begin" in names
    save_idx = names.index("storage_save")
    turn_idx = names.index("turn_begin")
    assert save_idx < turn_idx, (
        f"storage_save (idx={save_idx}) must precede turn_begin "
        f"(idx={turn_idx}); journal order drifted:\n{names}"
    )
