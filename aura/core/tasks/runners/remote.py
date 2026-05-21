"""RemoteAgentTask — subprocess teammate runner.

Translates claude-code's ``tasks/RemoteAgentTask`` into Python. The
runner spawns a fresh Python subprocess invoking
``python -m cli.teammate_entrypoint`` (or any equivalent module-level
entrypoint) and tracks its exit code. Communication with the parent is
exclusively via the on-disk JSONL mailbox under the shared storage
root — the subprocess never holds a back-channel IPC connection to the
parent, mirroring the constraints of the pane backend that originally
motivated the design.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

from aura.core.persistence import journal

#: Default Python module entrypoint used when ``module`` is omitted.
#: Matches the pane backend's invocation contract so a RemoteAgentTask
#: spawned from the runner package targets the same code path as the
#: existing pane-driven teammates.
DEFAULT_ENTRYPOINT_MODULE = "cli.teammate_entrypoint"


class RemoteAgentTask:
    """Subprocess runner — fires ``python -m <module>`` with the given args.

    The runner does not parse subprocess output; the spawned process is
    expected to communicate exclusively through the shared mailbox /
    storage layout. Exit code is surfaced via :attr:`returncode`; a
    non-zero exit journals a warning so operators can correlate
    subprocess failures with parent-side coordination errors.
    """

    def __init__(
        self,
        *,
        team_id: str,
        member_name: str,
        storage_root: Path | str,
        agent_type: str = "general-purpose",
        model_name: str | None = None,
        system_prompt: str | None = None,
        seed_prompt: str | None = None,
        module: str = DEFAULT_ENTRYPOINT_MODULE,
        extra_args: Sequence[str] = (),
        python_executable: str | None = None,
    ) -> None:
        self._team_id = team_id
        self._member_name = member_name
        self._storage_root = Path(storage_root)
        self._agent_type = agent_type
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._seed_prompt = seed_prompt
        self._module = module
        self._extra_args = list(extra_args)
        self._python = python_executable or sys.executable
        self._proc: asyncio.subprocess.Process | None = None
        self._wait_task: asyncio.Task[int] | None = None

    def _argv(self) -> list[str]:
        """Build the subprocess argv.

        Mirrors :func:`cli.teammate_entrypoint._make_parser`: every
        required flag is forwarded by name so the entrypoint stays
        callable by hand for debugging.
        """
        argv = [
            self._python,
            "-m",
            self._module,
            "--team-id",
            self._team_id,
            "--member",
            self._member_name,
            "--storage-root",
            str(self._storage_root),
            "--agent-type",
            self._agent_type,
        ]
        if self._model_name is not None:
            argv.extend(("--model", self._model_name))
        if self._system_prompt is not None:
            argv.extend(("--system-prompt", self._system_prompt))
        if self._seed_prompt is not None:
            argv.extend(("--seed-prompt", self._seed_prompt))
        argv.extend(self._extra_args)
        return argv

    async def start(self) -> asyncio.subprocess.Process:
        """Spawn the subprocess; return the Process handle.

        Idempotent — repeat calls return the already-running process.
        ``stdout`` / ``stderr`` are inherited from the parent so the
        operator sees teammate logs in the same terminal (matching the
        pane backend's interactive UX).
        """
        if self._proc is not None:
            return self._proc
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv(),
        )
        journal.write(
            "remote_agent_spawned",
            team_id=self._team_id,
            member=self._member_name,
            pid=self._proc.pid,
        )

        async def _wait() -> int:
            assert self._proc is not None
            code = await self._proc.wait()
            journal.write(
                "remote_agent_exited",
                team_id=self._team_id,
                member=self._member_name,
                pid=self._proc.pid,
                returncode=code,
            )
            return code

        self._wait_task = asyncio.create_task(
            _wait(),
            name=f"aura-remote-agent-wait-{self._member_name}",
        )
        return self._proc

    async def wait_for_terminal(self) -> int:
        """Await the subprocess; return its exit code (``-1`` if never started)."""
        if self._wait_task is None:
            return -1
        return await self._wait_task

    def abort(self) -> None:
        """Terminate the subprocess (SIGTERM). Idempotent.

        SIGTERM gives the entrypoint a chance to flush mailbox state
        and exit cleanly; the entrypoint maps ``KeyboardInterrupt`` /
        signal-induced exits to ``130`` (matching the main CLI). For a
        hard kill, the caller should compose ``abort()`` with a manual
        ``self._proc.kill()`` after a grace window.
        """
        if self._proc is None or self._proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            self._proc.terminate()

    @property
    def returncode(self) -> int | None:
        """Subprocess exit code; ``None`` while still running."""
        return self._proc.returncode if self._proc is not None else None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None


__all__ = ["DEFAULT_ENTRYPOINT_MODULE", "RemoteAgentTask"]
