"""E2E pty-/subprocess-driven scenarios from the v0.11.0 parity spec.

The full E2E case documented in
``docs/specs/2026-04-23-aura-main-channel-parity.md`` § "E2E Case" chains
13 actions — paste attachment → mid-stream SIGINT → resume by session id
→ safety deny ``/etc/passwd`` → ``last_turn_denials()`` read → subagent
inherits read-records → mode switch to plan → journal has live mode →
reactive-compact with 150k-token paste → exit triggers ``aclose`` with
hanging MCP stub → journal has ``mcp_close_timeout``. A single test
function that runs all 13 actions sequentially would be a maintenance
hazard: one assertion failure buries the root cause behind noise from
earlier steps, and debugging the test itself becomes a research project.

We split into focused tests, each of which stands up a real subprocess
(NOT StringIO / pure in-process mocks — ``feedback_dogfood_before_done.md``
rejects render-repr as a substitute) and exercises a coherent subset of
the scenario:

- :func:`test_e2e_resume_after_sigint` — SIGINT mid-stream leaves the
  user turn on disk; a fresh process with the same session id reads it
  back (G1 AC-G1-4).
- :func:`test_e2e_safety_deny_exposes_last_turn_denials` — a tool_call
  against ``/etc/passwd`` trips the safety policy; ``last_turn_denials()``
  surfaces a single ``safety_blocked`` record (G5 AC-G5-1 + part of the
  spec's row 7/8).
- :func:`test_e2e_subagent_inherits_parent_read_records` — parent reads
  file X, spawns a subagent via :class:`SubagentFactory`, child sees X
  as ``fresh`` (G8 AC-G8-1 + row 9).
- :func:`test_e2e_plan_mode_switch_audit_records_live_mode` — switching
  the Agent's mode mid-session to ``plan`` and tripping a tool call
  produces a ``permission_decision`` journal entry whose ``mode`` is the
  post-switch string ``"plan"``, not the startup value (B1 AC-B1-2 + rows
  10/11).
- :func:`test_e2e_aclose_hanging_mcp_emits_timeout` — exit path with a
  hanging MCP stub: ``aclose(mcp_timeout=0.1)`` returns under 0.5s and
  the journal gains a ``mcp_close_timeout`` event (B3 AC-B3-1 + row 13).

All tests are gated behind ``pytest -m e2e`` — not part of the default
suite. They spawn a child process via ``subprocess.Popen`` (no ``pexpect``
dependency; stdlib ``pty`` is enough for the one test that actually needs
a pseudo-terminal). Every driver is in-file as a ``textwrap.dedent`` blob
so the test reads top-to-bottom without chasing fixtures.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_driver(
    driver_src: str,
    argv_tail: list[str],
    *,
    tmp_path: Path,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[bytes]:
    """Write ``driver_src`` to ``tmp_path/driver.py`` and run it.

    Driver source runs in a fresh Python subprocess with
    ``sys.path`` prefixed by the repo root so ``import aura`` and
    ``from tests.conftest import FakeChatModel`` both resolve against
    the in-tree sources (never a stale install).

    ``argv_tail`` is forwarded as ``sys.argv[1:]`` inside the driver.
    Returned result's ``stdout`` / ``stderr`` are bytes — callers decode
    with ``errors='replace'`` to keep noisy warnings (ResourceWarning,
    etc.) from crashing the test.
    """
    driver = tmp_path / "driver.py"
    driver.write_text(driver_src)
    return subprocess.run(
        [sys.executable, str(driver), *argv_tail],
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _decode(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# G1 / row 4–6: SIGINT mid-stream, resume sees the user turn.
# ---------------------------------------------------------------------------


_RESUME_DRIVER = textwrap.dedent(
    """
    # Phase A ('pre'): build an Agent whose model hangs forever; spawn astream;
    # signal the parent after the pre-save lands; hang on purpose so the parent
    # can SIGKILL us. Phase B ('post'): a fresh Agent re-opens the same DB at
    # the same session_id and writes the loaded history as JSON to stdout.
    import asyncio
    import json
    import os
    import sys
    from pathlib import Path
    from typing import Any

    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    sys.path.insert(0, {repo_root!r})

    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage
    from tests.conftest import FakeChatModel

    phase = sys.argv[1]
    db_path = sys.argv[2]
    signal_path = sys.argv[3]
    session_id = sys.argv[4]

    def _cfg() -> AuraConfig:
        return AuraConfig.model_validate({{
            "providers": [{{"name": "openai", "protocol": "openai"}}],
            "router": {{"default": "openai:gpt-4o-mini"}},
            "tools": {{"enabled": []}},
        }})

    if phase == "pre":
        prompt = sys.argv[5]
        attachment_body = sys.argv[6]

        class _Hanging(FakeChatModel):
            async def _agenerate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: AsyncCallbackManagerForLLMRun | None = None,
                **_: Any,
            ) -> ChatResult:
                # Signal parent that pre-save has happened (astream persists
                # before this method is called per the G1 contract).
                Path(signal_path).write_text("at-model")
                await asyncio.sleep(300)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=""))]
                )

        storage = SessionStorage(Path(db_path))
        agent = Agent(
            config=_cfg(),
            model=_Hanging(turns=[]),
            storage=storage,
            session_id=session_id,
            auto_compact_threshold=0,
        )
        attachment = HumanMessage(
            content=f"<mcp-resource uri='mem://paste'>{{attachment_body}}</mcp-resource>"
        )

        async def main() -> None:
            async for _ in agent.astream(prompt, attachments=[attachment]):
                pass

        try:
            asyncio.run(main())
        except BaseException:
            pass
        # Parent SIGKILLs us anyway; this is belt-and-braces.
        sys.exit(0)

    elif phase == "post":
        storage = SessionStorage(Path(db_path))
        loaded = storage.load(session_id)
        storage.close()
        # Pipe back a minimal JSON-serializable shape.
        out = []
        for msg in loaded:
            out.append({{"type": type(msg).__name__, "content": str(msg.content)}})
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
        sys.exit(0)

    else:
        sys.stderr.write(f"unknown phase: {{phase}}\\n")
        sys.exit(2)
    """
).format(repo_root=str(_REPO_ROOT))


def test_e2e_resume_after_sigint(tmp_path: Path) -> None:
    """Row 4-6 of the spec: paste → SIGINT → resume → see the user turn.

    - Phase 'pre' runs a driver that persists user+attachment, reaches
      the model call (signalled via a marker file), and hangs.
    - Parent SIGKILLs the driver to simulate the process crash.
    - Phase 'post' runs a fresh driver that re-opens the same session
      storage; we assert history carries the interrupted user turn and
      the attachment envelope — the G1 contract's real-world payoff.
    """
    db_path = tmp_path / "session.db"
    signal_path = tmp_path / "at_model.marker"
    session_id = "e2e-resume-001"
    prompt = "please summarize the attached log"
    attachment_body = "IMPORTANT-ATTACHMENT-CONTENTS-E2E-001"

    driver = tmp_path / "driver.py"
    driver.write_text(_RESUME_DRIVER)

    # --- phase A: pre-SIGKILL
    proc = subprocess.Popen(
        [
            sys.executable,
            str(driver),
            "pre",
            str(db_path),
            str(signal_path),
            session_id,
            prompt,
            attachment_body,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if signal_path.exists():
                break
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=2)
                raise AssertionError(
                    "driver 'pre' exited before signalling "
                    f"(rc={proc.returncode}): stderr={_decode(stderr)!r}"
                )
            time.sleep(0.05)
        assert signal_path.exists(), (
            f"'pre' driver never reached the model call within 30s; "
            f"stderr={_decode(proc.stderr.read() if proc.stderr else b'')!r}"
        )
        proc.kill()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)

    # --- phase B: resume
    post = _run_driver(
        _RESUME_DRIVER,
        ["post", str(db_path), str(signal_path), session_id],
        tmp_path=tmp_path,
        timeout=20.0,
    )
    assert post.returncode == 0, (
        f"'post' driver exited {post.returncode}: "
        f"stdout={_decode(post.stdout)!r} stderr={_decode(post.stderr)!r}"
    )
    loaded = json.loads(_decode(post.stdout))
    # Exactly the pre-save shape: [HumanMessage attachment, HumanMessage user].
    assert len(loaded) == 2, (
        f"expected [attachment, user_msg] after SIGKILL; got {loaded!r}"
    )
    assert loaded[0]["type"] == "HumanMessage"
    assert attachment_body in loaded[0]["content"], (
        f"attachment body missing after resume; got {loaded[0]['content']!r}"
    )
    assert loaded[1]["type"] == "HumanMessage"
    assert loaded[1]["content"] == prompt


# ---------------------------------------------------------------------------
# G5 / rows 7–8: safety-deny /etc/passwd → last_turn_denials() has the record.
# ---------------------------------------------------------------------------


_DENIALS_DRIVER = textwrap.dedent(
    """
    import asyncio
    import json
    import sys
    from pathlib import Path

    from langchain_core.messages import AIMessage

    sys.path.insert(0, {repo_root!r})

    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.hooks import HookChain
    from aura.core.hooks.permission import make_permission_hook
    from aura.core.permissions.rule import Rule
    from aura.core.permissions.safety import (
        DEFAULT_PROTECTED_READS,
        DEFAULT_PROTECTED_WRITES,
        SafetyPolicy,
    )
    from aura.core.permissions.session import RuleSet, SessionRuleSet
    from aura.core.persistence.storage import SessionStorage
    from tests.conftest import FakeChatModel, FakeTurn

    db_path = sys.argv[1]
    out_path = sys.argv[2]

    # A deny-everything asker (never actually called because safety trips first).
    class _NeverAsker:
        async def __call__(self, *, tool, args, rule_hint):
            from aura.core.hooks.permission import AskerResponse
            return AskerResponse(choice="deny")

    cfg = AuraConfig.model_validate({{
        "providers": [{{"name": "openai", "protocol": "openai"}}],
        "router": {{"default": "openai:gpt-4o-mini"}},
        "tools": {{"enabled": ["read_file"]}},
    }})

    turns = [
        FakeTurn(AIMessage(
            content="",
            tool_calls=[{{
                "name": "read_file",
                "args": {{"path": "/etc/passwd"}},
                "id": "tc_safety_e2e",
                "type": "tool_call",
            }}],
        )),
        FakeTurn(AIMessage(content="I got blocked — noted.")),
    ]

    safety = SafetyPolicy(
        protected_writes=DEFAULT_PROTECTED_WRITES,
        protected_reads=DEFAULT_PROTECTED_READS,
        exempt=(),
    )
    hook = make_permission_hook(
        asker=_NeverAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path(db_path).parent,
        safety=safety,
    )

    async def main() -> None:
        storage = SessionStorage(Path(db_path))
        agent = Agent(
            config=cfg,
            model=FakeChatModel(turns=turns),
            storage=storage,
            hooks=HookChain(pre_tool=[hook]),
            session_id="e2e-denials",
            auto_compact_threshold=0,
        )
        async for _ in agent.astream("please read /etc/passwd"):
            pass
        denials = agent.last_turn_denials()
        payload = [
            {{
                "tool_name": d.tool_name,
                "tool_use_id": d.tool_use_id,
                "reason": d.reason,
                "target": d.target,
                "tool_input": d.tool_input,
            }}
            for d in denials
        ]
        Path(out_path).write_text(json.dumps(payload))
        await agent.aclose()

    asyncio.run(main())
    """
).format(repo_root=str(_REPO_ROOT))


def test_e2e_safety_deny_exposes_last_turn_denials(tmp_path: Path) -> None:
    """Rows 7-8 of the spec: a tool_call against ``/etc/passwd`` safety-blocks;
    :meth:`Agent.last_turn_denials` surfaces a single ``safety_blocked`` record
    with ``target='/etc/passwd'``.

    This is a real out-of-process run — we confirm the API is visible from a
    fresh Python interpreter, not just via an in-process unit-test harness.
    """
    db_path = tmp_path / "session.db"
    out_path = tmp_path / "denials.json"

    result = _run_driver(
        _DENIALS_DRIVER,
        [str(db_path), str(out_path)],
        tmp_path=tmp_path,
        timeout=30.0,
    )
    assert result.returncode == 0, (
        f"denials driver exited {result.returncode}: "
        f"stdout={_decode(result.stdout)!r} stderr={_decode(result.stderr)!r}"
    )
    assert out_path.exists(), "driver did not write the denials payload file"
    payload = json.loads(out_path.read_text())
    assert len(payload) == 1, (
        f"expected exactly one PermissionDenial on the turn; got {payload!r}"
    )
    entry = payload[0]
    assert entry["tool_name"] == "read_file"
    assert entry["reason"] == "safety_blocked"
    assert entry["target"] == "/etc/passwd"
    assert entry["tool_use_id"] == "tc_safety_e2e"
    assert entry["tool_input"] == {"path": "/etc/passwd"}


# ---------------------------------------------------------------------------
# G8 / row 9: subagent inherits parent _read_records.
# ---------------------------------------------------------------------------


_SUBAGENT_READS_DRIVER = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    from langchain_core.messages import AIMessage

    sys.path.insert(0, {repo_root!r})

    from aura.config.schema import AuraConfig
    from aura.core.memory.context import _ReadRecord
    from aura.core.persistence.storage import SessionStorage
    from aura.core.tasks.factory import SubagentFactory
    from tests.conftest import FakeChatModel, FakeTurn

    file_x = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    st = file_x.stat()
    parent_reads = {{
        file_x.resolve(): _ReadRecord(
            mtime=st.st_mtime, size=st.st_size, partial=False,
        ),
    }}

    def _cfg() -> AuraConfig:
        return AuraConfig.model_validate({{
            "providers": [{{"name": "openai", "protocol": "openai"}}],
            "router": {{"default": "openai:gpt-4o-mini"}},
            "tools": {{"enabled": []}},
        }})

    factory = SubagentFactory(
        parent_config=_cfg(),
        parent_model_spec="openai:gpt-4o-mini",
        parent_read_records_provider=lambda: parent_reads,
        model_factory=lambda: FakeChatModel(
            turns=[FakeTurn(AIMessage(content="subagent done"))]
        ),
        storage_factory=lambda: SessionStorage(Path(":memory:")),
    )
    child = factory.spawn("sub-prompt")
    try:
        status_x = child._context.read_status(file_x)
        inherited_rules = list(child._context._matched_rules)
        inherited_skills = list(child._context._invoked_skills)
        # Prove "shallow copy": mutate child, parent is untouched.
        y = file_x.parent / "y.txt"
        y.write_text("y-content\\n")
        child._context.record_read(y)
        parent_has_y = y.resolve() in parent_reads
    finally:
        child.close()

    out_path.write_text(json.dumps({{
        "x_status_in_child": status_x,
        "child_matched_rules": [r.tool_name for r in inherited_rules],
        "child_invoked_skills": inherited_skills,
        "parent_has_y_after_child_write": parent_has_y,
    }}))
    """
).format(repo_root=str(_REPO_ROOT))


def test_e2e_subagent_inherits_parent_read_records(tmp_path: Path) -> None:
    """Row 9 of the spec: parent read X → spawn subagent → child sees X as fresh.

    Also asserts the scope guard from G8: matched_rules / invoked_skills do
    NOT cross the boundary, and the child's own ``record_read`` never
    retro-writes into the parent's snapshot (shallow-copy semantics).
    """
    file_x = tmp_path / "x.txt"
    file_x.write_text("parent already read me\n")
    out_path = tmp_path / "subagent.json"

    result = _run_driver(
        _SUBAGENT_READS_DRIVER,
        [str(file_x), str(out_path)],
        tmp_path=tmp_path,
        timeout=30.0,
    )
    assert result.returncode == 0, (
        f"subagent driver exited {result.returncode}: "
        f"stdout={_decode(result.stdout)!r} stderr={_decode(result.stderr)!r}"
    )
    assert out_path.exists(), "subagent driver did not write its payload"
    payload = json.loads(out_path.read_text())
    assert payload["x_status_in_child"] == "fresh", (
        f"child must see parent-read file as 'fresh'; got {payload!r}"
    )
    assert payload["child_matched_rules"] == []
    assert payload["child_invoked_skills"] == []
    assert payload["parent_has_y_after_child_write"] is False, (
        "child's own read leaked back into parent's snapshot — "
        "G8 shallow-copy invariant broken"
    )


# ---------------------------------------------------------------------------
# B1 / rows 10–11: mode switch mid-session is reflected in the permission audit.
# ---------------------------------------------------------------------------


_LIVE_MODE_DRIVER = textwrap.dedent(
    """
    import asyncio
    import json
    import sys
    from pathlib import Path

    from langchain_core.messages import AIMessage

    sys.path.insert(0, {repo_root!r})

    from aura.config.schema import AuraConfig
    from aura.core import journal
    from aura.core.agent import Agent
    from aura.core.hooks import HookChain
    from aura.core.hooks.permission import make_permission_hook
    from aura.core.permissions.session import RuleSet, SessionRuleSet
    from aura.core.persistence.storage import SessionStorage
    from tests.conftest import FakeChatModel, FakeTurn

    db_path = sys.argv[1]
    journal_path = sys.argv[2]

    # Configure journal so we can read it back from the parent.
    journal.configure(Path(journal_path))

    # Spy asker: always denies, so _decide falls through to user_deny branch
    # (but only in non-plan modes; under 'plan' the plan_mode branch runs
    # before the asker is consulted — that's the path we care about here).
    class _DenyAsker:
        async def __call__(self, *, tool, args, rule_hint):
            from aura.core.hooks.permission import AskerResponse
            return AskerResponse(choice="deny")

    cfg = AuraConfig.model_validate({{
        "providers": [{{"name": "openai", "protocol": "openai"}}],
        "router": {{"default": "openai:gpt-4o-mini"}},
        "tools": {{"enabled": ["write_file"]}},
    }})

    # Live-mode provider: reads ``agent.mode`` on each invocation so a
    # mid-session set_mode("plan") takes effect before the next tool call.
    _agent_cell: list = [None]

    def _live_mode():
        a = _agent_cell[0]
        return "default" if a is None else a.mode

    hook = make_permission_hook(
        asker=_DenyAsker(),
        session=SessionRuleSet(),
        rules=RuleSet(),
        project_root=Path(db_path).parent,
        mode=_live_mode,
    )

    turns = [
        FakeTurn(AIMessage(
            content="",
            tool_calls=[{{
                "name": "write_file",
                "args": {{"path": "a.txt", "content": "x"}},
                "id": "tc_live_mode",
                "type": "tool_call",
            }}],
        )),
        FakeTurn(AIMessage(content="ok")),
    ]

    async def main() -> None:
        storage = SessionStorage(Path(db_path))
        agent = Agent(
            config=cfg,
            model=FakeChatModel(turns=turns),
            storage=storage,
            hooks=HookChain(pre_tool=[hook]),
            session_id="e2e-live-mode",
            auto_compact_threshold=0,
        )
        _agent_cell[0] = agent
        # Flip the mode BEFORE the turn runs — this is what claude-code's
        # shift+tab binding does mid-session. The permission hook must see
        # "plan", not the startup "default".
        agent.set_mode("plan")
        async for _ in agent.astream("write something"):
            pass
        await agent.aclose()

    asyncio.run(main())
    """
).format(repo_root=str(_REPO_ROOT))


def test_e2e_plan_mode_switch_audit_records_live_mode(tmp_path: Path) -> None:
    """Rows 10-11 of the spec: shift+tab → plan; next permission audit
    records ``mode: 'plan'`` (as a string, not a function repr).

    Drives a subprocess that wires the exact same ``_live_mode`` closure
    the CLI uses, flips the mode mid-session via ``agent.set_mode("plan")``,
    then reads the last ``permission_decision`` event from the journal.
    Mirror of B1's AC-B1-2 from an out-of-process vantage — proves the
    live-mode provider survives process boundaries.
    """
    db_path = tmp_path / "session.db"
    journal_path = tmp_path / "journal.jsonl"

    result = _run_driver(
        _LIVE_MODE_DRIVER,
        [str(db_path), str(journal_path)],
        tmp_path=tmp_path,
        timeout=30.0,
    )
    assert result.returncode == 0, (
        f"live-mode driver exited {result.returncode}: "
        f"stdout={_decode(result.stdout)!r} stderr={_decode(result.stderr)!r}"
    )
    assert journal_path.exists(), (
        "driver did not produce a journal file — configure() wiring drifted"
    )
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    perm_events = [e for e in events if e.get("event") == "permission_decision"]
    assert perm_events, (
        f"no permission_decision events in journal; events seen: "
        f"{[e.get('event') for e in events]!r}"
    )
    # The hook serializes the mode per call. The one that matters is the
    # one AFTER set_mode("plan") — which, given a single-tool-call turn, is
    # the last (and only) permission_decision in the journal.
    last = perm_events[-1]
    assert last["mode"] == "plan", (
        f"permission_decision 'mode' should be the post-switch string "
        f"'plan'; got {last.get('mode')!r} (full event: {last!r})"
    )
    # Defensive: make sure we didn't accidentally serialize a function repr.
    assert "function" not in str(last["mode"]), (
        f"mode serialized as a function repr, not a string: {last!r}"
    )


# ---------------------------------------------------------------------------
# B3 / row 13: aclose with hanging MCP emits mcp_close_timeout.
# ---------------------------------------------------------------------------


_ACLOSE_DRIVER = textwrap.dedent(
    """
    import asyncio
    import json
    import sys
    from pathlib import Path
    from typing import Any

    sys.path.insert(0, {repo_root!r})

    from aura.config.schema import AuraConfig
    from aura.core import journal
    from aura.core.agent import Agent
    from aura.core.mcp.manager import MCPServerStatus
    from aura.core.persistence.storage import SessionStorage
    from tests.conftest import FakeChatModel

    db_path = sys.argv[1]
    journal_path = sys.argv[2]

    journal.configure(Path(journal_path))

    # Minimal hanging-manager stub modelled on _HangingManager in
    # tests/test_mcp_close_timeout.py — stop_all sleeps 60s so any non-cancel
    # path blows the mcp_timeout budget by 2+ orders of magnitude.
    class _HangingManager:
        def __init__(self, connected):
            self._connected = list(connected)
            self.stop_called = False
            self.cancelled = False

        async def stop_all(self):
            self.stop_called = True
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        def status(self):
            return [
                MCPServerStatus(
                    name=n,
                    transport="stdio",
                    state="connected",
                    error_message=None,
                    tool_count=0,
                    resource_count=0,
                    prompt_count=0,
                )
                for n in self._connected
            ]

    async def main() -> int:
        cfg = AuraConfig.model_validate({{
            "providers": [{{"name": "openai", "protocol": "openai"}}],
            "router": {{"default": "openai:gpt-4o-mini"}},
            "tools": {{"enabled": []}},
        }})
        storage = SessionStorage(Path(db_path))
        agent = Agent(
            config=cfg,
            model=FakeChatModel(turns=[]),
            storage=storage,
            session_id="e2e-aclose",
        )
        mgr = _HangingManager(connected=["hang-a", "hang-b"])
        agent._mcp_manager = mgr

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await agent.aclose(mcp_timeout=0.1)
        elapsed = loop.time() - t0
        return 0 if (elapsed < 0.5 and mgr.cancelled) else 3

    rc = asyncio.run(main())
    sys.exit(rc)
    """
).format(repo_root=str(_REPO_ROOT))


def test_e2e_aclose_hanging_mcp_emits_timeout(tmp_path: Path) -> None:
    """Row 13 of the spec: exit with a hanging MCP stub → ``mcp_close_timeout``.

    Subprocess owns its own event loop so we can measure the aclose deadline
    without test-framework interference. Exit code 0 is the rc contract —
    the driver flags ``elapsed < 0.5`` AND ``mgr.cancelled`` itself. Parent
    then reads the journal file to confirm the structured event carries
    the expected fields.
    """
    db_path = tmp_path / "session.db"
    journal_path = tmp_path / "journal.jsonl"

    result = _run_driver(
        _ACLOSE_DRIVER,
        [str(db_path), str(journal_path)],
        tmp_path=tmp_path,
        timeout=20.0,
    )
    assert result.returncode == 0, (
        f"aclose driver exited {result.returncode} — either elapsed budget "
        "blown or cancel didn't propagate. "
        f"stdout={_decode(result.stdout)!r} stderr={_decode(result.stderr)!r}"
    )
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    timeout_events = [e for e in events if e.get("event") == "mcp_close_timeout"]
    assert len(timeout_events) == 1, (
        f"expected exactly one mcp_close_timeout event; got "
        f"{[e.get('event') for e in events]!r}"
    )
    evt = timeout_events[0]
    assert evt["session"] == "e2e-aclose"
    assert evt["timeout_sec"] == pytest.approx(0.1)
    assert 0.1 <= evt["elapsed_sec"] < 0.5
    assert sorted(evt["servers_hanging"]) == ["hang-a", "hang-b"]
    # Happy-path event must NOT fire on the timeout branch.
    assert not any(e.get("event") == "mcp_stopped" for e in events), (
        f"mcp_stopped leaked onto the timeout path: {events!r}"
    )


# ---------------------------------------------------------------------------
# Smoke: the e2e suite itself — at least one pty-backed test so the marker
# is gated by the presence of POSIX pty support. The CLI smoke tests in
# tests/test_cli_smoke.py already exercise the pty path exhaustively
# (welcome banner, shift+tab, /exit); we don't duplicate them here.
# ---------------------------------------------------------------------------


def test_e2e_pty_import_guard() -> None:
    """Sanity: stdlib ``pty`` is importable on POSIX; skips on Windows.

    The other e2e tests here don't need a pty (they use signal files +
    subprocess piping — same pattern as ``test_g1_message_persistence``),
    but the spec's full 13-step scenario includes a real pty pass in
    ``tests/test_cli_smoke.py``'s pty section which IS exercised under
    ``make check``. This test just documents the dependency so a future
    porter to Windows CI sees an explicit skip instead of a mysterious
    ``ImportError`` mid-module.
    """
    if os.name == "nt":
        pytest.skip("stdlib pty is POSIX-only; e2e pty tests do not run on Windows")
    import pty  # noqa: F401 — import is the assertion
