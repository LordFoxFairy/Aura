"""PaneBackend — teammate runs as a subprocess inside a tmux pane.

Implements claude-code's pane backend (``utils/swarm/teamHelpers.ts``):

1. ``tmux split-window -h -P -F '#{pane_id}'`` to allocate a new pane;
   capture its ID.
2. ``tmux send-keys -t <pane_id> 'python -m aura.cli.teammate_entrypoint
   --team-id ... --member ... --storage-root ...' Enter`` to start the
   teammate subprocess inside that pane.
3. The subprocess loads its own :class:`~aura.core.agent.Agent`, builds a
   :class:`~aura.core.teams.mailbox.Mailbox`, and runs the same
   ``run_teammate`` loop — communication with the leader stays via the
   filesystem-rooted JSONL mailbox + ``.seen`` cursor that already works
   cross-process.

Shutdown contract:

- Graceful: post a ``shutdown_request`` to the teammate's inbox via the
  manager. The runtime picks it up at its next mailbox poll, emits
  ``shutdown_response`` to the leader, and exits. We then run
  ``tmux kill-pane -t <pane_id>`` to reclaim the visual real-estate.
- Force-kill: skip the wait; ``tmux kill-pane`` directly. The pane's
  shell receives SIGHUP, the Python subprocess gets SIGTERM, and the
  on-disk state stays consistent because every mailbox write is atomic.

We invoke ``tmux`` via :func:`subprocess.run` with a 5-second timeout —
no extra dependency, no third-party libtmux abstraction. Errors raise
:class:`PaneBackendError` so the manager can surface them as a
:class:`~aura.core.teams.manager.TeamError`.
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

from aura.core.persistence import journal
from aura.core.teams.backends.detection import pane_backend_available
from aura.core.teams.backends.types import BackendHandle
from aura.core.teams.types import (
    TEAM_LEADER_NAME,
    TeamMessage,
)

if TYPE_CHECKING:
    from aura.core.abort import AbortController
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage
    from aura.core.teams.manager import TeamManager
    from aura.core.teams.types import BackendType, TeammateMember


_TMUX_TIMEOUT_SEC: float = 5.0


class PaneBackendError(RuntimeError):
    """Raised when a tmux command fails or the environment lacks tmux."""


def _run_tmux(args: list[str]) -> str:
    """Run ``tmux <args>`` and return stripped stdout.

    Raises :class:`PaneBackendError` on non-zero exit, missing binary,
    or timeout. Stdout/stderr are captured so the diagnostic message
    surfaces cleanly through the manager's :class:`TeamError` path.
    """
    cmd = ["tmux", *args]
    try:
        proc = subprocess.run(  # noqa: S603 — fixed binary, validated args.
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
    """Return ``True`` iff ``pane_id`` exists in the current tmux server.

    Implemented with ``tmux list-panes -a -F '#{pane_id}'`` and a
    membership check; cheap (single IPC) and survives a server restart
    by simply reporting "no" (the pane is, in fact, gone).
    """
    try:
        out = _run_tmux(["list-panes", "-a", "-F", "#{pane_id}"])
    except PaneBackendError:
        return False
    return pane_id in out.split()


@dataclass
class PaneHandle(BackendHandle):
    """Handle for a teammate running inside a tmux pane.

    ``manager`` is held by reference so :meth:`shutdown` can post a
    ``shutdown_request`` through the same code path the in-process
    backend uses. ``stop_event`` mirrors the in-process API for
    interface compatibility; the subprocess does NOT see this event
    directly (it lives in a different memory space) but we still
    fire it so any in-leader observers (tests, future hooks) match
    the in-process behaviour.
    """

    pane_id: str | None
    member_name: str
    manager: TeamManager
    stop_event: asyncio.Event
    abort: AbortController

    async def shutdown(self, *, timeout_sec: float = 5.0) -> bool:
        """Cooperative stop via mailbox shutdown_request, then kill the pane.

        Returns ``True`` when the teammate emitted a matching
        ``shutdown_response`` on the leader's inbox within
        ``timeout_sec`` (i.e. the subprocess exited cleanly);
        ``False`` when the wait timed out and we force-killed.
        """
        if self.pane_id is None or not _pane_alive(self.pane_id):
            return True
        # Snapshot leader-mailbox baseline BEFORE posting the request so
        # we don't false-positive on a stale ack from a previous run.
        team = self.manager.team
        if team is None:
            await self.force_kill()
            return False
        from aura.core.teams.mailbox import Mailbox  # local import; cycle-safe.
        mailbox = Mailbox(self.manager._storage, team.team_id)  # noqa: SLF001
        baseline = {m.msg_id for m in mailbox.read_all(TEAM_LEADER_NAME)}
        # Post shutdown_request via the manager's internal poster so the
        # message is observed by the same journal events the in-process
        # path emits (parity for /tasks + observability).
        with contextlib.suppress(Exception):
            self.manager._post(  # noqa: SLF001
                TeamMessage(
                    msg_id=uuid.uuid4().hex,
                    sender=TEAM_LEADER_NAME,
                    recipient=self.member_name,
                    body="shutdown",
                    kind="shutdown_request",
                ),
            )
        self.stop_event.set()
        # Off-thread poll, mirroring the in-process aremove_member path.
        # Same 50ms cadence as the manager's internal waiter so the
        # combined latency is bounded by the runtime's mailbox poll.
        acked = await asyncio.to_thread(
            self._wait_for_ack, mailbox, baseline, timeout_sec,
        )
        await self._kill_pane()
        return acked

    def _wait_for_ack(self, mailbox: object, baseline: set[str], timeout: float) -> bool:
        """Block until a matching shutdown_response lands or timeout.

        ``mailbox`` typed loose to dodge the ``TYPE_CHECKING`` import.
        Same shape as
        :meth:`TeamManager._wait_for_shutdown_response`.
        """
        import time as _time
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
        """Kill the pane immediately; abort the controller for parity."""
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
            # Pane may already be gone — that's a success state for us.
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
    """Singleton pane backend.

    Construction does NOT validate the environment — the registry runs
    :func:`~aura.core.teams.backends.detection.pane_backend_available`
    before handing the singleton out. ``spawn`` re-checks defensively
    so a programmatic instantiation surfaces the same error path.
    """

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
    ) -> PaneHandle:
        """Split a pane and start the teammate subprocess inside it.

        ``agent`` is unused here — the subprocess builds its own Agent
        from the storage root. We accept it to match the Protocol so
        the manager can dispatch uniformly. ``seed_prompt`` is forwarded
        to the subprocess via ``--seed-prompt`` and consumed there.
        """
        del agent  # subprocess builds its own
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
        # Persist the pane_id onto the member BEFORE we launch the
        # subprocess so a crash between split + send-keys still leaves
        # config.json with a recoverable handle. Pydantic models are
        # mutable in-place; the manager will _persist() right after.
        member.tmux_pane_id = pane_id
        cmd = self._build_subprocess_command(
            team_id=team_id,
            member=member,
            storage=storage,
            seed_prompt=seed_prompt,
        )
        # send-keys with " Enter" submits the line in the pane's shell.
        # Quoting via shlex.join is critical: paths or seed prompts may
        # contain spaces, and tmux passes the literal string to the shell.
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
        """Build the ``python -m aura.cli.teammate_entrypoint ...`` argv.

        Uses ``sys.executable`` so the spawned subprocess inherits the
        same interpreter (venv + aura installed). ``storage_root`` is
        the on-disk parent of the leader's storage path so the teammate
        finds the same teams/ directory layout.
        """
        storage_root = _resolve_storage_root(storage)
        argv = [
            sys.executable,
            "-m",
            "aura.cli.teammate_entrypoint",
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
    """Return the storage-root path the subprocess should pass.

    ``storage._path`` is the index sqlite file (e.g.
    ``~/.aura/index.sqlite``); the subprocess wants the directory
    containing it so its own ``SessionStorage`` constructor can wire
    teams/ + projects/ underneath. For ``:memory:`` storage (tests),
    fall back to the default Aura root so we don't pass a literal
    ``:memory:`` to the subprocess (which would re-init an empty,
    invisible-to-the-leader sqlite).
    """
    db_path = getattr(storage, "_path", None)
    if db_path is None or str(db_path) == ":memory:":
        return os.path.expanduser("~/.aura")
    return str(db_path.parent)


__all__ = ["PaneBackend", "PaneBackendError", "PaneHandle"]
