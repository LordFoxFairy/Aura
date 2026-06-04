"""Read-only projection of a team into a TeamViewSnapshot for the ``/team`` surface."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable

from aura.application.tasks.store import TasksStore
from aura.application.teams.mailbox import Mailbox
from aura.application.teams.state import Member, TeamError
from aura.application.teams.view_types import TeammateMemberStatus, TeamViewSnapshot
from aura.domain.team import TEAM_LEADER_NAME, TeamMessage, TeamRecord
from aura.infrastructure.persistence.storage import SessionStorage

_RECENT_MESSAGE_CAP: int = 10


class TeamViewBuilder:
    def __init__(
        self,
        *,
        team: Callable[[], TeamRecord | None],
        members: dict[str, Member],
        storage: SessionStorage,
        tasks_store: TasksStore,
    ) -> None:
        self._team = team
        self._members = members
        self._storage = storage
        self._tasks_store = tasks_store

    def view_state(self, team_id: str | None = None) -> TeamViewSnapshot:
        live = self._team()
        if team_id is None:
            if live is None:
                raise TeamError("no team is active; pass team_id explicitly")
            record = live
        elif live is not None and live.team_id == team_id:
            record = live
        else:
            # Off-record snapshot reloads config.json so concurrent writers are visible.
            path = self._storage.team_config_path(team_id)
            if not path.exists():
                raise TeamError(f"team {team_id!r} not found on disk")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise TeamError(
                    f"team {team_id!r} config is unreadable: {exc}",
                ) from exc
            record = TeamRecord.model_validate(raw)
        members = self._build_member_statuses(record)
        recent = self._collect_recent_messages(record)
        sub_count, tx_count = self._count_artifacts(record)
        return TeamViewSnapshot(
            team_id=record.team_id,
            name=record.name,
            members=members,
            recent_messages=recent,
            subagent_count=sub_count,
            transcript_count=tx_count,
        )

    def _build_member_statuses(
        self, record: TeamRecord,
    ) -> list[TeammateMemberStatus]:
        out: list[TeammateMemberStatus] = []
        team = self._team()
        live = team is not None and team.team_id == record.team_id
        for m in record.members:
            tokens = 0
            last_active: float | None = None
            model_spec: str | None = m.model_name
            slot = self._members.get(m.name)
            if live and slot is not None and slot.task_id is not None:
                rec = self._tasks_store.get(slot.task_id)
                if rec is not None:
                    tokens = int(rec.progress.token_count)
                    last_active = rec.progress.last_activity_at
                    # Resolved spec reflects the inherited default when override is empty.
                    if rec.model_spec:
                        model_spec = rec.model_spec
            shutting_down = slot is not None and slot.shutdown_waiter is not None
            if not m.is_active:
                status = "dead"
            elif live and shutting_down:
                status = "shutting-down"
            else:
                status = "active"
            out.append(
                TeammateMemberStatus(
                    name=m.name,
                    agent_type=m.agent_type,
                    model_spec=model_spec,
                    status=status,
                    tokens_used=tokens,
                    last_active=last_active,
                    lifecycle_state=(
                        "unknown" if slot is None
                        else slot.lifecycle_state or "unknown"
                    ),
                ),
            )
        return out

    def _collect_recent_messages(
        self, record: TeamRecord,
    ) -> list[TeamMessage]:
        mailbox = Mailbox(self._storage, record.team_id)
        recipients = [TEAM_LEADER_NAME] + [m.name for m in record.members]
        gathered: list[TeamMessage] = []
        for rcpt in recipients:
            gathered.extend(mailbox.read_all(rcpt))
        gathered.sort(key=lambda m: m.sent_at, reverse=True)
        return gathered[:_RECENT_MESSAGE_CAP]

    def _count_artifacts(self, record: TeamRecord) -> tuple[int, int]:
        sub_count = 0
        with contextlib.suppress(Exception):
            sub_count = len(self._storage.list_subagent_transcripts())
        tx_count = 0
        try:
            tx_dir = self._storage.team_root(record.team_id) / "transcripts"
            if tx_dir.is_dir():
                tx_count = sum(
                    1 for p in tx_dir.iterdir()
                    if p.is_file() and p.suffix == ".jsonl"
                )
        except OSError:
            tx_count = 0
        return sub_count, tx_count
