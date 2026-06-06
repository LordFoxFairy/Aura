"""SendMessage tool — outside-team error, recipient validation, fan-out."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from aura.application.session import AgentSession
from aura.application.tasks.spawn import SpawnContext, SubagentSpawner
from aura.application.tasks.store import TasksStore
from aura.application.teams.manager import TeamManager
from aura.application.teams.state import TeamError
from aura.application.teams.team_port import TeamPort
from aura.config.schema import AuraConfig
from aura.domain.permission.safety import DEFAULT_SAFETY
from aura.domain.permission.session import RuleSet
from aura.domain.team import (
    BROADCAST_RECIPIENT,
    MAX_BODY_CHARS,
    TEAM_LEADER_NAME,
    TeamMessage,
    TeamMessageKind,
    TeamRecord,
)
from aura.domain.tool import ToolError
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.events import CoordinationEvent
from aura.tools.send_message import SendMessage, SendMessageParams, _preview
from tests.conftest import FakeChatModel, FakeTurn


def _cfg() -> AuraConfig:
    # Without ``teams.enabled=True`` the spawned teammate's ``join_team`` raises.
    return AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "teams": {"enabled": True},
    })


def _factory() -> SubagentSpawner[AgentSession]:
    return SubagentSpawner(
        SpawnContext(
            parent_config=_cfg(),
            parent_model_spec="openai:gpt-4o-mini",
            build_child=AgentSession,
            parent_ruleset=RuleSet(),
            parent_safety=DEFAULT_SAFETY,
            parent_mode_provider=lambda: "default",
            model_factory=lambda: FakeChatModel(
                turns=[FakeTurn(AIMessage(content="ack"))],
            ),
            storage_factory=lambda: SessionStorage(Path(":memory:")),
        )
    )


async def _no_runtime(**_kwargs: Any) -> None:
    return


def _leader(storage: SessionStorage) -> Any:
    leader = MagicMock()
    leader.session_id = "leader-1"
    leader.cwd = Path.cwd()
    leader._storage = storage
    leader.join_team = MagicMock()
    leader.leave_team = MagicMock()
    return leader


@pytest.mark.asyncio
async def test_send_message_outside_team_raises(tmp_path: Path) -> None:
    agent = MagicMock()
    agent.team = None
    tool = SendMessage(
        team_provider=lambda: agent.team,
        member_name_provider=lambda: agent._team_member_name,
    )
    with pytest.raises(ToolError, match="not in a team"):
        await tool._arun(to="alice", body="hi")


@pytest.mark.asyncio
async def test_send_message_unknown_recipient_raises(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    leader.team = mgr
    tool = SendMessage(
        team_provider=lambda: leader.team,
        member_name_provider=lambda: leader._team_member_name,
    )
    with pytest.raises(ToolError, match="unknown recipient"):
        await tool._arun(to="ghost", body="hi")


@pytest.mark.asyncio
async def test_send_message_to_member_routes_to_mailbox(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    leader.team = mgr
    leader._team_member_name = None  # leader has no member name
    tool = SendMessage(
        team_provider=lambda: leader.team,
        member_name_provider=lambda: leader._team_member_name,
    )
    result = await tool._arun(to="alice", body="please scan")
    assert result["recipient"] == "alice"
    assert result["sender"] == "leader"
    assert result["fanout"] == 1
    inbox = mgr.mailbox().read_all("alice")
    assert [m.body for m in inbox] == ["please scan"]


@pytest.mark.asyncio
async def test_send_message_sender_is_team_member_when_set(tmp_path: Path) -> None:
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    teammate = MagicMock()
    teammate.team = mgr
    teammate._team_member_name = "alice"
    tool = SendMessage(
        team_provider=lambda: teammate.team,
        member_name_provider=lambda: teammate._team_member_name,
    )
    result = await tool._arun(to="leader", body="here is my report")
    assert result["sender"] == "alice"
    assert result["recipient"] == "leader"


class _StubTeam:
    """Minimal TeamPort-shaped seam: lets us drive ``record is None`` and the
    ``manager.send`` ValueError→ToolError conversion without real infra. Unused
    TeamPort members raise so an accidental call is loud, not silent."""

    def __init__(
        self,
        *,
        active: bool,
        record: TeamRecord | None,
        send_error: Exception | None = None,
        sent: list[TeamMessage] | None = None,
    ) -> None:
        self._active = active
        self._record = record
        self._send_error = send_error
        self._sent = sent if sent is not None else []
        self.calls: list[dict[str, str]] = []

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def team(self) -> TeamRecord | None:
        return self._record

    @property
    def storage(self) -> SessionStorage:
        raise NotImplementedError

    def post_message(self, msg: TeamMessage) -> None:
        raise NotImplementedError

    def send(
        self,
        *,
        sender: str,
        recipient: str,
        body: str,
        kind: TeamMessageKind = "text",
    ) -> list[TeamMessage]:
        self.calls.append(
            {"sender": sender, "recipient": recipient, "body": body, "kind": kind},
        )
        if self._send_error is not None:
            raise self._send_error
        return self._sent

    def confirm_shutdown(self, member_name: str, *, body: str = "") -> None:
        raise NotImplementedError

    @property
    def pending_protocol_events(self) -> tuple[CoordinationEvent, ...]:
        raise NotImplementedError

    def drain_protocol_events(self) -> list[CoordinationEvent]:
        raise NotImplementedError

    async def cleanup_session_teams(self) -> None:
        raise NotImplementedError


def _build_tool(stub: _StubTeam, member_name: str | None = None) -> SendMessage:
    def provider() -> TeamPort | None:
        return stub

    return SendMessage(
        team_provider=provider,
        member_name_provider=lambda: member_name,
    )


def _msg(recipient: str, *, sender: str = "leader", body: str = "x") -> TeamMessage:
    return TeamMessage(msg_id="m1", sender=sender, recipient=recipient, body=body)


# --- provider-seam error branches -----------------------------------------


@pytest.mark.asyncio
async def test_send_message_inactive_manager_raises() -> None:
    """A bound-but-torn-down team (manager present, is_active False) must still
    reject sends so a member cannot mail into a dead team."""
    stub = _StubTeam(active=False, record=None)
    tool = _build_tool(stub)
    with pytest.raises(ToolError, match="not in a team"):
        await tool._arun(to="leader", body="hi")
    assert stub.calls == []  # never reached send()


@pytest.mark.asyncio
async def test_send_message_active_but_no_record_allows_leader_only() -> None:
    """Active team with a null record: only leader/broadcast are valid names,
    so member routing fails closed even though the manager is live."""
    stub = _StubTeam(active=True, record=None)
    tool = _build_tool(stub)
    with pytest.raises(ToolError, match="unknown recipient 'alice'"):
        await tool._arun(to="alice", body="hi")


@pytest.mark.asyncio
async def test_send_message_active_no_record_leader_passes_to_send() -> None:
    """With a null record the leader is always a valid recipient; send() is the
    sole authority and its result shapes the tool payload."""
    stub = _StubTeam(active=True, record=None, sent=[_msg("leader")])
    tool = _build_tool(stub)
    result = await tool._arun(to=TEAM_LEADER_NAME, body="report")
    assert result["recipient"] == TEAM_LEADER_NAME
    assert result["fanout"] == 1
    assert stub.calls[0]["recipient"] == TEAM_LEADER_NAME


@pytest.mark.asyncio
async def test_send_message_team_error_becomes_tool_error() -> None:
    """manager.send raising TeamError (a ValueError) must surface as a clean
    ToolError, not leak the domain exception to the model."""
    stub = _StubTeam(
        active=True,
        record=None,
        send_error=TeamError("body must contain at least one non-whitespace char"),
    )
    tool = _build_tool(stub)
    with pytest.raises(ToolError, match="non-whitespace") as exc:
        await tool._arun(to="leader", body="   ")
    assert isinstance(exc.value.__cause__, TeamError)


@pytest.mark.asyncio
async def test_send_message_plain_value_error_becomes_tool_error() -> None:
    """Any ValueError from the send seam (not just TeamError) is normalised to
    ToolError so the tool boundary never raises a raw ValueError."""
    stub = _StubTeam(active=True, record=None, send_error=ValueError("boom"))
    tool = _build_tool(stub)
    with pytest.raises(ToolError, match="send_message: boom"):
        await tool._arun(to="leader", body="hi")


@pytest.mark.asyncio
async def test_send_message_non_value_error_propagates_unwrapped() -> None:
    """Errors outside the ValueError family (e.g. RuntimeError) are NOT swallowed
    into ToolError — only ValueError is caught — so real faults stay visible."""
    stub = _StubTeam(active=True, record=None, send_error=RuntimeError("io down"))
    tool = _build_tool(stub)
    with pytest.raises(RuntimeError, match="io down"):
        await tool._arun(to="leader", body="hi")


@pytest.mark.asyncio
async def test_send_message_sender_defaults_to_leader_when_provider_none() -> None:
    """A null member-name provider means the caller is the leader; the recorded
    sender must fall back to the canonical leader name."""
    stub = _StubTeam(active=True, record=None, sent=[_msg("leader")])
    tool = _build_tool(stub, member_name=None)
    result = await tool._arun(to=TEAM_LEADER_NAME, body="hi")
    assert result["sender"] == TEAM_LEADER_NAME
    assert stub.calls[0]["sender"] == TEAM_LEADER_NAME


@pytest.mark.asyncio
async def test_send_message_kind_is_forwarded_to_send() -> None:
    """The shutdown_request kind must reach the manager verbatim; the tool is a
    pass-through for message semantics, not a rewriter."""
    stub = _StubTeam(active=True, record=None, sent=[_msg("leader")])
    tool = _build_tool(stub)
    await tool._arun(to=TEAM_LEADER_NAME, body="please exit", kind="shutdown_request")
    assert stub.calls[0]["kind"] == "shutdown_request"


# --- real-manager broadcast fan-out ---------------------------------------


@pytest.mark.asyncio
async def test_send_message_broadcast_fans_out_to_all_members(tmp_path: Path) -> None:
    """Broadcast must deliver one copy per member; fanout reflects the real
    recipient count so the model learns how wide the message went."""
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    mgr.add_member("bob")
    leader.team = mgr
    leader._team_member_name = None
    tool = SendMessage(
        team_provider=lambda: leader.team,
        member_name_provider=lambda: leader._team_member_name,
    )
    result = await tool._arun(to=BROADCAST_RECIPIENT, body="standup now")
    assert result["recipient"] == BROADCAST_RECIPIENT
    assert result["fanout"] == 2
    assert [m.body for m in mgr.mailbox().read_all("alice")] == ["standup now"]
    assert [m.body for m in mgr.mailbox().read_all("bob")] == ["standup now"]


@pytest.mark.asyncio
async def test_send_message_broadcast_empty_team_errors(tmp_path: Path) -> None:
    """Broadcasting into a memberless team is a ValueError from send and must
    convert to a ToolError, never a silent zero-fanout success."""
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    leader.team = mgr
    leader._team_member_name = None
    tool = SendMessage(
        team_provider=lambda: leader.team,
        member_name_provider=lambda: leader._team_member_name,
    )
    with pytest.raises(ToolError, match="no members"):
        await tool._arun(to=BROADCAST_RECIPIENT, body="anyone?")


@pytest.mark.asyncio
async def test_send_message_idempotent_double_send_appends_twice(
    tmp_path: Path,
) -> None:
    """The tool is append-only, not deduplicating: calling it twice with the
    same body must yield two distinct mailbox entries and two msg_ids."""
    storage = SessionStorage(tmp_path / "sessions.db")
    leader = _leader(storage)
    mgr = TeamManager(
        leader=leader,
        storage=storage,
        factory=_factory(),
        running_aborts={},
        tasks_store=TasksStore(),
        runtime_runner=_no_runtime,
    )
    mgr.create_team("alpha")
    mgr.add_member("alice")
    leader.team = mgr
    leader._team_member_name = None
    tool = SendMessage(
        team_provider=lambda: leader.team,
        member_name_provider=lambda: leader._team_member_name,
    )
    first = await tool._arun(to="alice", body="ping")
    second = await tool._arun(to="alice", body="ping")
    assert first["msg_id"] != second["msg_id"]
    assert [m.body for m in mgr.mailbox().read_all("alice")] == ["ping", "ping"]


# --- sync guard -----------------------------------------------------------


def test_send_message_sync_run_is_unsupported() -> None:
    """The tool is async-only; the sync entrypoint must fail loudly so no caller
    silently runs a no-op blocking path."""
    tool = _build_tool(_StubTeam(active=True, record=None))
    with pytest.raises(NotImplementedError, match="async-only"):
        tool._run(to="leader", body="hi")


# --- preview (args_preview) -----------------------------------------------


def test_preview_truncates_and_flattens_body() -> None:
    """args_preview is a one-line UI summary: it must cap the body at 40 chars
    and collapse newlines so multi-line sends don't break the status row."""
    preview = _preview({"to": "alice", "body": "line1\nline2" + "x" * 60})
    assert preview.startswith("send_message → alice: ")
    assert "\n" not in preview
    assert len(preview.split(": ", 1)[1]) == 40


def test_preview_tolerates_missing_keys() -> None:
    """Preview runs on raw, possibly-incomplete tool args; missing to/body must
    not crash the renderer — it falls back to a '?' recipient and empty body."""
    assert _preview({}) == "send_message → ?: "


# --- SendMessageParams schema-crash matrix --------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("to", ""),
        ("to", "x" * 65),
        ("body", ""),
        ("body", "x" * (MAX_BODY_CHARS + 1)),
    ],
)
def test_params_rejects_out_of_range_lengths(field: str, value: str) -> None:
    """Length guards bound the wire surface: empty/over-long to or body must be
    rejected by Pydantic before the tool ever touches the mailbox."""
    payload: dict[str, str] = {"to": "alice", "body": "hi"}
    payload[field] = value
    with pytest.raises(ValidationError):
        SendMessageParams.model_validate(payload)


def test_params_forbids_unknown_fields() -> None:
    """extra='forbid' blocks field injection so a hallucinated key cannot smuggle
    state past the schema boundary."""
    with pytest.raises(ValidationError):
        SendMessageParams.model_validate({"to": "a", "body": "b", "evil": 1})


@pytest.mark.parametrize("missing", ["to", "body"])
def test_params_requires_to_and_body(missing: str) -> None:
    """Both to and body are mandatory; omitting either is a schema crash, not a
    silently-defaulted empty send."""
    payload: dict[str, str] = {"to": "alice", "body": "hi"}
    del payload[missing]
    with pytest.raises(ValidationError):
        SendMessageParams.model_validate(payload)


def test_params_rejects_invalid_kind() -> None:
    """kind is a closed Literal; an off-menu value (e.g. shutdown_response, which
    is a manager-only reply kind) must not be accepted as an inbound send kind."""
    with pytest.raises(ValidationError):
        SendMessageParams.model_validate(
            {"to": "a", "body": "b", "kind": "shutdown_response"},
        )


def test_params_defaults_kind_to_text() -> None:
    """Omitting kind yields the 'text' default so normal sends need no kind."""
    params = SendMessageParams.model_validate({"to": "a", "body": "b"})
    assert params.kind == "text"
