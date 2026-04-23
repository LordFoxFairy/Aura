"""/status, /diff, /log — git-aware slash commands.

Quick, sub-second shell-outs to ``git`` that avoid a full model-turn
round-trip when the user just wants to know "what's dirty", "what
changed", or "what did I just commit". They share a tiny ``_git``
async helper that handles three repeatable failure modes (git not
installed, timeout, non-git-dir) so each command's ``handle`` body
stays focused on its own formatting.

Output strategy:
- ``/status`` formats output with rich markup and returns it as
  ``result.text`` (the REPL's console prints and interprets markup).
- ``/diff`` and ``/log`` emit ANSI-coloured git output by writing
  directly to an injected ``writer`` (defaulting to ``sys.stdout.write``),
  then return an empty ``text`` so the REPL's ``if result.text:`` guard
  skips auto-printing. Going straight to ``stdout`` keeps git's SGR
  codes intact without a rich/``Text.from_ansi`` round-trip — and,
  importantly, keeps ``aura/core/**`` free of the ``rich`` dependency
  (see ``test_core_does_not_import_ui_frameworks``).

The optional ``writer`` ctor arg is a seam for tests — they pass
``StringIO().write`` and assert on the captured buffer.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from aura.core.commands.types import CommandResult, CommandSource

if TYPE_CHECKING:
    from aura.core.agent import Agent


Writer = Callable[[str], object]


_DEFAULT_TIMEOUT_S = 5.0
_DIFF_MAX_LINES = 500
_LOG_MIN = 1
_LOG_MAX = 100
_LOG_DEFAULT = 20


# ---------------------------------------------------------------------------
# Shared shell-out helper
# ---------------------------------------------------------------------------


class _GitNotInstalledError(RuntimeError):
    """Raised when the ``git`` binary is missing from PATH."""


class _GitTimeoutError(RuntimeError):
    """Raised when a git subprocess exceeds its wall-clock budget."""


async def _git(
    *args: str,
    cwd: Path,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> tuple[int, str, str]:
    """Run ``git <args>`` asynchronously and return (exit, stdout, stderr).

    Raises:
        _GitNotInstalledError: ``git`` binary missing from PATH.
        _GitTimeoutError: child did not complete within ``timeout_s``.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise _GitNotInstalledError(str(exc)) from exc

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except TimeoutError as exc:
        # Kill rather than terminate — we want the child gone *now*, and
        # the user's next slash-command shouldn't inherit its pipes.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        # Drain pipes so we don't leak fds; ignore any further output.
        with contextlib.suppress(Exception):
            await proc.wait()
        raise _GitTimeoutError(
            f"git timed out after {timeout_s:.0f}s",
        ) from exc

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    return proc.returncode or 0, stdout, stderr


def _not_a_repo(stderr: str) -> bool:
    msg = stderr.lower()
    return "not a git repository" in msg or "fatal: not a git" in msg


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


class GitStatusCommand:
    """``/status`` — short ``git status`` with coloured file list."""

    name = "/status"
    description = "git status (short + branch)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = None

    def __init__(self, *, writer: Writer | None = None) -> None:
        # Writer is kept for symmetry with /diff and /log; /status
        # renders via the CommandResult text so the REPL's printer owns
        # the output. Accepting the arg keeps the constructor shape
        # uniform across the three commands.
        self._writer = writer

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
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

        return CommandResult(
            handled=True, kind="print", text=_format_status(stdout),
        )


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
    head = parts[0]
    tail = parts[1] if len(parts) == 2 else ""
    branch = head.split("...", 1)[0]

    extras: list[str] = []
    if "[" in tail and tail.endswith("]"):
        bracket = tail[tail.index("[") + 1 : -1]
        for piece in bracket.split(","):
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


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------


class GitDiffCommand:
    """``/diff [--full|--staged]`` — coloured diff summary or full patch."""

    name = "/diff"
    description = "git diff (stat by default; --full or --staged)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[--full|--staged]"

    def __init__(self, *, writer: Writer | None = None) -> None:
        self._writer = writer

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
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
                handled=True, kind="print",
                text="[dim]no changes[/dim]",
            )

        self._print_ansi(stdout)
        return CommandResult(handled=True, kind="print", text="")

    def _print_ansi(self, stdout: str) -> None:
        write = self._writer or sys.stdout.write
        lines = stdout.splitlines()
        truncated = len(lines) > _DIFF_MAX_LINES
        shown = lines[:_DIFF_MAX_LINES] if truncated else lines
        # Write the ANSI stream verbatim: the terminal owns SGR
        # interpretation, so we don't need to parse/re-emit it. The
        # trailing newline matters — without it the truncation marker
        # would collide with the last diff line.
        write("\n".join(shown))
        write("\n")
        if truncated:
            # Keep the marker plaintext — it's dim-intent but we can't
            # use rich markup here (no rich import in core/**). The CLI
            # renders it as an unadorned dim-grey line, which is fine.
            write(
                "\x1b[2m… truncated (use --full or run `git diff` in "
                "a shell)\x1b[0m\n"
            )


# ---------------------------------------------------------------------------
# /log
# ---------------------------------------------------------------------------


class GitLogCommand:
    """``/log [N]`` — last N commits (default 20, clamped to [1, 100])."""

    name = "/log"
    description = "git log --oneline (default 20, up to 100)"
    source: CommandSource = "builtin"
    allowed_tools: tuple[str, ...] = ()
    argument_hint: str | None = "[N]"

    def __init__(self, *, writer: Writer | None = None) -> None:
        self._writer = writer

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
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
                handled=True, kind="print",
                text="[dim]no commits yet[/dim]",
            )

        write = self._writer or sys.stdout.write
        # Preserve trailing newline so the next REPL prompt lands on a
        # fresh line instead of butting up against the last SHA.
        body = stdout if stdout.endswith("\n") else stdout + "\n"
        write(body)
        return CommandResult(handled=True, kind="print", text="")


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


# ---------------------------------------------------------------------------
# Shared error-result factories
# ---------------------------------------------------------------------------


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
