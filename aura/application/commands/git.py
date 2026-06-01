"""/status, /diff, /log — git-aware slash commands.

Sub-second shell-outs to ``git``. ``/status`` returns rich-markup text;
``/diff`` and ``/log`` emit ANSI-coloured output directly to ``stdout``
(or an injected test writer) so git's SGR codes survive without a rich
round-trip — and ``aura/core/**`` stays UI-framework-free.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path

from aura.application.commands.types import AgentT, CommandResult, CommandSource

Writer = Callable[[str], object]


_DEFAULT_TIMEOUT_S = 5.0
_DIFF_MAX_LINES = 500
_LOG_MIN = 1
_LOG_MAX = 100
_LOG_DEFAULT = 20


class _GitNotInstalledError(RuntimeError):
    """``git`` binary missing from PATH."""


class _GitTimeoutError(RuntimeError):
    """git subprocess exceeded its wall-clock budget."""


async def _git(
    *args: str,
    cwd: Path,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[int, str, str]:
    """Run ``git <args>`` async; return ``(exit, stdout, stderr)``."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise _GitNotInstalledError(str(exc)) from exc
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except TimeoutError as exc:
        # kill (not terminate) — child must be gone before the next slash cmd.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise _GitTimeoutError(f"git timed out after {timeout_s:.0f}s") from exc
    return (
        proc.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


def _not_a_repo(stderr: str) -> bool:
    msg = stderr.lower()
    return "not a git repository" in msg or "fatal: not a git" in msg


class GitStatusCommand:
    """``/status`` — short ``git status`` with coloured file list."""

    name = "/status"
    description = "git status (short + branch)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    def __init__(self, *, writer: Writer | None = None) -> None:
        # Symmetry with /diff /log; /status renders via CommandResult.text, never writes here.
        self._writer = writer

    async def handle(self, arg: str, agent: AgentT) -> CommandResult:
        cwd = Path.cwd()
        try:
            code, stdout, stderr = await _git(
                "status", "--short", "--branch", cwd=cwd,
            )
        except _GitNotInstalledError:
            return _not_installed_result()
        except _GitTimeoutError:
            return _timeout_result("/status")

        if code != 0:
            if _not_a_repo(stderr):
                return CommandResult(
                    handled=True, kind="print",
                    text="error: not a git repository",
                )
            return CommandResult(
                handled=True, kind="print",
                text=f"[dim]{stderr.strip() or 'git status failed'}[/dim]",
            )
        return CommandResult(handled=True, kind="view", text=_format_status(stdout))


def _format_status(raw: str) -> str:
    lines = raw.splitlines()
    if not lines:
        return "[dim]working tree clean[/dim]"

    out: list[str] = []
    branch_line = lines[0] if lines[0].startswith("##") else None
    file_lines = lines[1:] if branch_line else lines

    if branch_line is not None:
        out.append(_format_branch_line(branch_line))

    if not file_lines:
        out.append("[dim]working tree clean[/dim]")
        return "\n".join(out)

    for line in file_lines:
        out.append(_format_file_line(line))
    return "\n".join(out)


def _format_branch_line(line: str) -> str:
    """Render the ``##`` header as ``on branch X · N ahead · N behind``."""
    body = line[3:] if line.startswith("## ") else line.lstrip("#").strip()
    parts = body.split(" ", 1)
    branch = parts[0].split("...", 1)[0]
    tail = parts[1] if len(parts) == 2 else ""
    extras: list[str] = []
    if "[" in tail and tail.endswith("]"):
        for piece in tail[tail.index("[") + 1 : -1].split(","):
            piece = piece.strip()
            if piece.startswith("ahead "):
                extras.append(f"{piece[6:]} ahead")
            elif piece.startswith("behind "):
                extras.append(f"{piece[7:]} behind")
            elif piece.startswith("gone"):
                extras.append("upstream gone")
    suffix = (" · " + " · ".join(extras)) if extras else ""
    return f"[bold cyan]on branch {branch}[/bold cyan]{suffix}"


_FILE_STYLES: dict[str, str] = {
    "??": "dim",
    " M": "yellow",
    "M ": "yellow",
    "MM": "yellow",
    " A": "green",
    "A ": "green",
    "AM": "green",
    " D": "red",
    "D ": "red",
    "R ": "green",
    "RM": "green",
    "C ": "green",
    " U": "red",
    "UU": "red",
    "!!": "dim",
}


def _format_file_line(line: str) -> str:
    if len(line) < 3:
        return line
    code = line[:2]
    style = _FILE_STYLES.get(code, "dim")
    return f"[{style}]{line}[/{style}]"


class GitDiffCommand:
    """``/diff [--full|--staged]`` — coloured diff summary or full patch."""

    name = "/diff"
    description = "git diff (stat by default; --full or --staged)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[--full|--staged]"

    def __init__(self, *, writer: Writer | None = None) -> None:
        self._writer = writer

    async def handle(self, arg: str, agent: AgentT) -> CommandResult:
        flags = arg.split()
        full = "--full" in flags
        staged = "--staged" in flags
        for tok in flags:
            if tok not in {"--full", "--staged"}:
                return CommandResult(
                    handled=True, kind="print",
                    text=f"error: unknown flag {tok!r}",
                )

        git_args: list[str] = ["diff", "--color=always"]
        if staged:
            git_args.append("--staged")
        if not full:
            git_args.append("--stat")

        cwd = Path.cwd()
        try:
            code, stdout, stderr = await _git(*git_args, cwd=cwd)
        except _GitNotInstalledError:
            return _not_installed_result()
        except _GitTimeoutError:
            return _timeout_result("/diff")

        if code != 0:
            if _not_a_repo(stderr):
                return CommandResult(
                    handled=True, kind="print",
                    text="error: not a git repository",
                )
            return CommandResult(
                handled=True, kind="print",
                text=f"[dim]{stderr.strip() or 'git diff failed'}[/dim]",
            )

        if not stdout.strip():
            return CommandResult(
                handled=True, kind="print", text="[dim]no changes[/dim]",
            )
        # Diff printed to stdout already; empty-text view triggers REPL pause.
        self._print_ansi(stdout)
        return CommandResult(handled=True, kind="view", text="")

    def _print_ansi(self, stdout: str) -> None:
        write = self._writer or sys.stdout.write
        lines = stdout.splitlines()
        truncated = len(lines) > _DIFF_MAX_LINES
        shown = lines[:_DIFF_MAX_LINES] if truncated else lines
        write("\n".join(shown))
        write("\n")
        if truncated:
            write(
                "\x1b[2m… truncated (use --full or run `git diff` in "
                "a shell)\x1b[0m\n"
            )


class GitLogCommand:
    """``/log [N]`` — last N commits (default 20, clamped to [1, 100])."""

    name = "/log"
    description = "git log --oneline (default 20, up to 100)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[N]"

    def __init__(self, *, writer: Writer | None = None) -> None:
        self._writer = writer

    async def handle(self, arg: str, agent: AgentT) -> CommandResult:
        n = _parse_log_count(arg)
        if isinstance(n, str):
            return CommandResult(handled=True, kind="print", text=n)

        cwd = Path.cwd()
        try:
            code, stdout, stderr = await _git(
                "log", "--oneline", "--decorate", "--color=always",
                f"-{n}", cwd=cwd,
            )
        except _GitNotInstalledError:
            return _not_installed_result()
        except _GitTimeoutError:
            return _timeout_result("/log")

        if code != 0:
            if _not_a_repo(stderr):
                return CommandResult(
                    handled=True, kind="print",
                    text="error: not a git repository",
                )
            low = stderr.lower()
            if (
                "does not have any commits" in low
                or "bad default revision" in low
                or "unknown revision" in low
            ):
                return CommandResult(
                    handled=True, kind="print",
                    text="[dim]no commits yet[/dim]",
                )
            return CommandResult(
                handled=True, kind="print",
                text=f"[dim]{stderr.strip() or 'git log failed'}[/dim]",
            )

        if not stdout.strip():
            return CommandResult(
                handled=True, kind="print", text="[dim]no commits yet[/dim]",
            )
        write = self._writer or sys.stdout.write
        write(stdout if stdout.endswith("\n") else stdout + "\n")
        return CommandResult(handled=True, kind="view", text="")


def _parse_log_count(arg: str) -> int | str:
    arg = arg.strip()
    if not arg:
        return _LOG_DEFAULT
    try:
        n = int(arg)
    except ValueError:
        return f"error: /log takes an integer count, got {arg!r}"
    if n < _LOG_MIN:
        return _LOG_MIN
    if n > _LOG_MAX:
        return _LOG_MAX
    return n


def _not_installed_result() -> CommandResult:
    return CommandResult(
        handled=True, kind="print",
        text="error: git CLI not installed (install git to use "
             "/status /diff /log)",
    )


def _timeout_result(cmd_name: str) -> CommandResult:
    return CommandResult(
        handled=True, kind="print",
        text=(
            f"error: {cmd_name} timed out after "
            f"{_DEFAULT_TIMEOUT_S:.0f}s; run the command manually"
        ),
    )
