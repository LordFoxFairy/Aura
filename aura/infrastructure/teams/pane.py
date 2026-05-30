"""PaneBackend — teammate runs as a ``python -m cli teammate`` subprocess inside a tmux pane.

Shutdown contract:

- graceful: post ``shutdown_request`` to the inbox, await ``shutdown_response``, kill pane
- force: ``tmux kill-pane`` directly; mailbox writes are atomic so on-disk state stays consistent
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aura.domain.team import (
    TEAM_LEADER_NAME,
    TeamMessage,
)
from aura.infrastructure.persistence import journal
from aura.infrastructure.teams.detection import pane_backend_available
from aura.infrastructure.teams.types import BackendHandle

if TYPE_CHECKING:
    from aura.application.teams.mailbox import MailboxNotifier
    from aura.application.teams.manager import TeamManager
    from aura.core.agent import Agent
    from aura.domain.abort import AbortController
    from aura.domain.team import BackendType, TeammateMember
    from aura.infrastructure.persistence.storage import SessionStorage


_TMUX_TIMEOUT_SEC: float = 5.0


class PaneBackendError(RuntimeError):
    pass


def _run_tmux(args: list[str]) -> str:
    """Run ``tmux <args>`` and return stripped stdout; raises on non-zero / missing / timeout."""
    cmd = ["tmux", *args]
    try:
        proc = subprocess.run(  # noqa: S603  # fixed binary, callers pass validated args
            cmd,
            capture_output=True,
            text=True,
            timeout=_TMUX_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PaneBackendError("tmux binary not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise PaneBackendError(
            f"tmux command timed out after {_TMUX_TIMEOUT_SEC}s: "
            f"{shlex.join(cmd)}",
        ) from exc
    if proc.returncode != 0:
        raise PaneBackendError(
            f"tmux failed (rc={proc.returncode}): {shlex.join(cmd)}\n"
            f"stderr: {proc.stderr.strip()}",
        )
    return proc.stdout.strip()


def _pane_alive(pane_id: str) -> bool:
    try:
        out = _run_tmux(["list-panes", "-a", "-F", "#{pane_id}"])
    except PaneBackendError:
        return False
    return pane_id in out.split()


@dataclass
class PaneHandle(BackendHandle):
    pane_id: str | None
    member_name: str
    # Held to post shutdown_request through the same code path the in-process backend uses.
    manager: TeamManager
    # Subprocess can't see this event directly; we still fire it for in-leader observers.
    stop_event: asyncio.Event
    abort: AbortController

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        """Cooperative stop via mailbox; True on clean ack within timeout, else force-kill."""
        if self.pane_id is None or not _pane_alive(self.pane_id):
            return True
        # Snapshot baseline BEFORE posting so a stale ack from a previous run can't false-positive.
        team = self.manager.team
        if team is None:
            await self.force_kill()
            return False
        # Local import dodges the application↔infrastructure cycle.
        from aura.application.teams.mailbox import Mailbox  # noqa: PLC0415
        mailbox = Mailbox(self.manager.storage, team.team_id)
        baseline = {m.msg_id for m in mailbox.read_all(TEAM_LEADER_NAME)}
        # Reuse the manager's poster so journal events match the in-process path.
        with contextlib.suppress(Exception):
            self.manager.post_message(
                TeamMessage(
                    msg_id=uuid.uuid4().hex,
                    sender=TEAM_LEADER_NAME,
                    recipient=self.member_name,
                    body="shutdown",
                    kind="shutdown_request",
                ),
            )
        self.stop_event.set()
        # 50ms cadence matches the manager's internal waiter — bounds total latency by mailbox poll.
        acked = await asyncio.to_thread(
            self._wait_for_ack, mailbox, baseline, timeout_sec,
        )
        await self._kill_pane()
        return acked

    def _wait_for_ack(self, mailbox: object, baseline: set[str], timeout: float) -> bool:
        # JSONL on disk is the IPC channel — the subprocess can't share an asyncio.Future.
        import time as _time  # noqa: PLC0415  # keep ad-hoc poll local
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            for msg in mailbox.read_all(TEAM_LEADER_NAME):  # type: ignore[attr-defined]
                if msg.msg_id in baseline:
                    continue
                if (
                    msg.sender == self.member_name
                    and msg.kind == "shutdown_response"
                ):
                    return True
            _time.sleep(0.05)
        return False

    async def force_kill(self) -> None:
        if not self.abort.aborted:
            with contextlib.suppress(Exception):
                self.abort.abort("pane_force_kill")
        await self._kill_pane()

    async def _kill_pane(self) -> None:
        if self.pane_id is None:
            return
        try:
            await asyncio.to_thread(
                _run_tmux, ["kill-pane", "-t", self.pane_id],
            )
        except PaneBackendError as exc:
            # Pane already gone is a success state — still journal for forensics.
            journal.write(
                "team_pane_kill_error",
                pane_id=self.pane_id,
                error=str(exc),
            )

    def is_alive(self) -> bool:
        if self.pane_id is None:
            return False
        return _pane_alive(self.pane_id)


class PaneBackend:
    backend_type: BackendType = "pane"

    async def spawn(
        self,
        *,
        team_id: str,
        member: TeammateMember,
        agent: Agent,
        manager: TeamManager,
        storage: SessionStorage,
        stop_event: asyncio.Event,
        abort: AbortController,
        seed_prompt: str | None,
        notifier: MailboxNotifier | None = None,
    ) -> PaneHandle:
        """Split a pane, start the teammate subprocess; ``seed_prompt`` forwarded via CLI flag."""
        del agent  # subprocess builds its own Agent
        del notifier  # cross-process — asyncio queues can't span Python processes
        if not pane_backend_available():
            raise PaneBackendError(
                "pane backend requires tmux on PATH and an active tmux "
                "session ($TMUX must be set)",
            )
        pane_id = await asyncio.to_thread(
            _run_tmux,
            ["split-window", "-h", "-P", "-F", "#{pane_id}"],
        )
        if not pane_id:
            raise PaneBackendError(
                "tmux split-window returned no pane_id",
            )
        # Persist pane_id BEFORE send-keys so a crash mid-spawn still leaves a recoverable handle.
        member.tmux_pane_id = pane_id
        cmd = self._build_subprocess_command(
            team_id=team_id,
            member=member,
            storage=storage,
            seed_prompt=seed_prompt,
        )
        # shlex.join is required — tmux passes the literal string to the shell, spaces would split.
        await asyncio.to_thread(
            _run_tmux,
            ["send-keys", "-t", pane_id, shlex.join(cmd), "Enter"],
        )
        journal.write(
            "team_pane_spawned",
            team_id=team_id,
            member=member.name,
            pane_id=pane_id,
        )
        return PaneHandle(
            pane_id=pane_id,
            member_name=member.name,
            manager=manager,
            stop_event=stop_event,
            abort=abort,
        )

    @staticmethod
    def _build_subprocess_command(
        *,
        team_id: str,
        member: TeammateMember,
        storage: SessionStorage,
        seed_prompt: str | None,
    ) -> list[str]:
        # sys.executable so the subprocess inherits the same venv (aura installed).
        storage_root = _resolve_storage_root(storage)
        argv = [
            sys.executable,
            "-m",
            "cli",
            "teammate",
            "--team-id", team_id,
            "--member", member.name,
            "--storage-root", str(storage_root),
            "--agent-type", member.agent_type,
        ]
        if member.model_name:
            argv.extend(["--model", member.model_name])
        if member.system_prompt:
            argv.extend(["--system-prompt", member.system_prompt])
        if seed_prompt and seed_prompt.strip():
            argv.extend(["--seed-prompt", seed_prompt])
        return argv


def _resolve_storage_root(storage: SessionStorage) -> str:
    db_path = storage.path
    if str(db_path) == ":memory:":
        return os.path.expanduser("~/.aura")
    return str(db_path.parent)


__all__ = ["PaneBackend", "PaneBackendError", "PaneHandle"]
