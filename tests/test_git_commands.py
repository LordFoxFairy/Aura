"""Tests for the ``/status``, ``/diff``, ``/log`` slash commands.

Every test that exercises the happy path shells out to the real ``git``
binary via a per-test ``tmp_path`` fixture that ``git init``s a fresh
repo. Mocking ``asyncio.create_subprocess_exec`` would have lied about
the porcelain format — git's short-status line ordering and decorate
phrasing is exactly what we're parsing here, so testing against the real
thing is the only way to catch a regression before users do.

The one exception is the timeout path, which mocks the subprocess to
avoid a 5-second sleep in the suite.
"""

from __future__ import annotations

import asyncio
import io
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aura.application.commands.factory import build_default_registry
from aura.application.commands.git import (
    GitDiffCommand,
    GitLogCommand,
    GitStatusCommand,
    _format_branch_line,
    _format_file_line,
    _format_status,
    _git,
    _GitNotInstalledError,
    _GitTimeoutError,
    _parse_log_count,
)
from aura.application.session import AgentSession


def _run_git(cwd: Path, *args: str) -> None:
    """Run a sync git command for test setup; raise on failure."""
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _init_repo(cwd: Path) -> None:
    """Initialise a minimal repo with a deterministic identity + main branch."""
    _run_git(cwd, "init", "-q", "-b", "main")
    _run_git(cwd, "config", "user.email", "test@example.com")
    _run_git(cwd, "config", "user.name", "Test User")
    _run_git(cwd, "config", "commit.gpgsign", "false")


def _commit(cwd: Path, path: str, content: str, msg: str) -> None:
    (cwd / path).write_text(content)
    _run_git(cwd, "add", path)
    _run_git(cwd, "commit", "-q", "-m", msg)


def _capture_buf() -> io.StringIO:
    """Fresh buffer + ``.write``-compatible closure used as the command writer."""
    return io.StringIO()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fresh git repo as the process ``cwd`` for this test."""
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def agent() -> AgentSession:
    """MagicMock AgentSession — none of our commands touch agent state."""
    return MagicMock(spec=AgentSession)


def test_git_commands_registered_in_default_registry() -> None:
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert "/status" in names
    assert "/diff" in names
    assert "/log" in names


@pytest.mark.asyncio
async def test_status_clean_tree_prints_working_tree_clean(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "hello\n", "initial")

    result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "working tree clean" in result.text
    assert "on branch main" in result.text


@pytest.mark.asyncio
async def test_status_with_modified_files_shows_them(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "hello\n", "initial")
    (repo / "a.txt").write_text("changed\n")
    (repo / "new.txt").write_text("x\n")

    result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "a.txt" in result.text
    assert "new.txt" in result.text
    # Untracked file carries the ?? code, modified carries M — we don't
    # assert the rich-markup escapes literally (too brittle) but we do
    # verify the codes made it through.
    assert "??" in result.text
    assert "M" in result.text


@pytest.mark.asyncio
async def test_status_not_a_repo_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: AgentSession,
) -> None:
    # Plain tmp dir, NO git init — bare cwd outside any repo.
    non_repo = tmp_path / "plain"
    non_repo.mkdir()
    monkeypatch.chdir(non_repo)
    # Ensure we aren't accidentally inside a parent-dir git repo (e.g.
    # the Aura checkout itself). GIT_CEILING_DIRECTORIES blocks git's
    # upward walk past the ceiling.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "not a git repository" in result.text


@pytest.mark.asyncio
async def test_diff_default_shows_stat_summary(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "one\ntwo\n", "initial")
    (repo / "a.txt").write_text("one\nchanged\n")

    console = _capture_buf()
    result = await GitDiffCommand(writer=console.write).handle("", agent)

    assert result.handled is True
    # Direct-print path: text is empty, actual output is on the console.
    assert result.text == ""
    out = console.getvalue()
    # ``--stat`` produces "a.txt |" + changes summary.
    assert "a.txt" in out
    assert "|" in out


@pytest.mark.asyncio
async def test_diff_full_shows_patch_body(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "one\ntwo\n", "initial")
    (repo / "a.txt").write_text("one\nchanged\n")

    console = _capture_buf()
    result = await GitDiffCommand(writer=console.write).handle("--full", agent)

    assert result.handled is True
    out = console.getvalue()
    # Patch body carries the ``---``/``+++`` headers and the added line.
    assert "---" in out
    assert "+++" in out
    assert "changed" in out


@pytest.mark.asyncio
async def test_diff_staged_shows_index_diff(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "one\n", "initial")
    (repo / "a.txt").write_text("one\ntwo\n")
    _run_git(repo, "add", "a.txt")
    # Worktree now matches index, but index has a staged change. Plain
    # /diff (worktree vs index) must be empty; /diff --staged must show
    # the staged change.
    buf_worktree = _capture_buf()
    result_worktree = await GitDiffCommand(
        writer=buf_worktree.write,
    ).handle("", agent)
    assert "no changes" in result_worktree.text

    buf_staged = _capture_buf()
    result_staged = await GitDiffCommand(
        writer=buf_staged.write,
    ).handle("--staged", agent)
    assert result_staged.handled is True
    out = buf_staged.getvalue()
    assert "a.txt" in out


@pytest.mark.asyncio
async def test_diff_rejects_unknown_flag(agent: AgentSession) -> None:
    result = await GitDiffCommand().handle("--bogus", agent)
    assert result.handled is True
    assert result.text.startswith("error:")
    assert "--bogus" in result.text


@pytest.mark.asyncio
async def test_diff_truncates_at_500_lines(
    repo: Path, agent: AgentSession,
) -> None:
    # Seed with a 1000-line file, then rewrite it to trigger a huge diff.
    _commit(repo, "a.txt", "".join(f"{i}\n" for i in range(1000)), "seed")
    (repo / "a.txt").write_text(
        "".join(f"changed-{i}\n" for i in range(1000)),
    )

    console = _capture_buf()
    result = await GitDiffCommand(writer=console.write).handle("--full", agent)

    assert result.handled is True
    out = console.getvalue()
    assert "truncated" in out


@pytest.mark.asyncio
async def test_log_empty_repo_prints_no_commits_yet(
    repo: Path, agent: AgentSession,
) -> None:
    result = await GitLogCommand().handle("", agent)

    assert result.handled is True
    assert "no commits yet" in result.text


@pytest.mark.asyncio
async def test_log_with_commits_shows_them(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "v1\n", "first")
    _commit(repo, "a.txt", "v2\n", "second")
    _commit(repo, "a.txt", "v3\n", "third")

    console = _capture_buf()
    result = await GitLogCommand(writer=console.write).handle("", agent)

    assert result.handled is True
    out = console.getvalue()
    assert "first" in out
    assert "second" in out
    assert "third" in out


@pytest.mark.asyncio
async def test_log_respects_explicit_limit(
    repo: Path, agent: AgentSession,
) -> None:
    for i in range(10):
        _commit(repo, "a.txt", f"v{i}\n", f"msg-number-{i:02d}-zzz")

    console = _capture_buf()
    result = await GitLogCommand(writer=console.write).handle("3", agent)

    assert result.handled is True
    out = console.getvalue()
    # Only the 3 most recent commit messages should appear. Use
    # "zzz"-suffixed tags so substring checks can't collide with the
    # short SHA prefixes git prints alongside each line.
    assert "msg-number-09-zzz" in out
    assert "msg-number-08-zzz" in out
    assert "msg-number-07-zzz" in out
    assert "msg-number-00-zzz" not in out
    assert "msg-number-06-zzz" not in out


@pytest.mark.asyncio
async def test_log_clamps_count_above_100(
    repo: Path, agent: AgentSession,
) -> None:
    _commit(repo, "a.txt", "v1\n", "only")

    # Pass 200 — we can't easily observe the clamp in the output (only
    # one commit exists), so we patch ``_git`` and capture args instead.
    captured: dict[str, Any] = {}

    async def fake_git(
        *args: str, cwd: Path, timeout_s: float = 5.0,
    ) -> tuple[int, str, str]:
        captured["args"] = args
        return 0, "abc123 only\n", ""

    with patch(
        "aura.application.commands.git._git", side_effect=fake_git,
    ):
        await GitLogCommand().handle("200", agent)

    # Verify -100 made it in (not -200).
    assert "-100" in captured["args"]
    assert "-200" not in captured["args"]


@pytest.mark.asyncio
async def test_log_rejects_non_integer_arg(agent: AgentSession) -> None:
    result = await GitLogCommand().handle("abc", agent)
    assert result.handled is True
    assert result.text.startswith("error:")


@pytest.mark.asyncio
async def test_git_helper_raises_timeout(tmp_path: Path) -> None:
    # Build a fake subprocess whose ``.communicate()`` never completes
    # until wait_for trips. ``proc.kill()`` must be safe.
    async def never_completes() -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        return b"", b""

    fake_proc = MagicMock()
    fake_proc.communicate = never_completes
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock(return_value=0)
    fake_proc.returncode = None

    async def fake_exec(*_a: Any, **_kw: Any) -> Any:
        return fake_proc

    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        pytest.raises(_GitTimeoutError),
    ):
        await _git("status", cwd=tmp_path, timeout_s=0.05)


@pytest.mark.asyncio
async def test_status_timeout_returns_friendly_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: AgentSession,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def never_completes() -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        return b"", b""

    fake_proc = MagicMock()
    fake_proc.communicate = never_completes
    fake_proc.kill = MagicMock()
    fake_proc.wait = AsyncMock(return_value=0)
    fake_proc.returncode = None

    async def fake_exec(*_a: Any, **_kw: Any) -> Any:
        return fake_proc

    with (
        patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
        patch("aura.application.commands.git._DEFAULT_TIMEOUT_S", 0.05),
    ):
        result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "timed out" in result.text


@pytest.mark.asyncio
async def test_git_not_installed_returns_friendly_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: AgentSession,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def fake_exec(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError(2, "No such file", "git")

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "git CLI not installed" in result.text


def _stub_git(
    code: int, stdout: str, stderr: str,
) -> Callable[..., Any]:
    """Build a ``_git`` replacement returning a fixed ``(code, out, err)``.

    Mocks at the subprocess seam so generic-failure / not-a-repo branches
    are exercised without shelling out — git would never naturally emit
    an arbitrary nonzero stderr on a healthy repo.
    """

    async def fake(*_a: str, **_kw: Any) -> tuple[int, str, str]:
        return code, stdout, stderr

    return fake


def _raise_git(exc: BaseException) -> Callable[..., Any]:
    """Build a ``_git`` replacement that raises ``exc`` on call."""

    async def fake(*_a: str, **_kw: Any) -> tuple[int, str, str]:
        raise exc

    return fake


# --- _parse_log_count: numeric-boundary matrix --------------------------


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("0", 1),       # below min clamps up to _LOG_MIN
        ("-1", 1),      # negative clamps up, never a negative -N flag
        ("-9999", 1),   # extreme negative still clamps to min
        ("1", 1),       # exact lower bound passes through
        ("100", 100),   # exact upper bound passes through
        ("101", 100),   # above max clamps down to _LOG_MAX
        ("999999", 100),  # extreme positive clamps to max
        ("", 20),       # empty arg falls back to _LOG_DEFAULT
        ("   ", 20),    # whitespace-only treated as empty
        ("  7  ", 7),   # surrounding whitespace is stripped
    ],
)
def test_parse_log_count_clamps_to_valid_window(
    arg: str, expected: int,
) -> None:
    """/log N must clamp into [1,100] so we never pass git a bogus -N."""
    assert _parse_log_count(arg) == expected


@pytest.mark.parametrize(
    "arg",
    ["abc", "1.5", "0x10", "nan", "inf", "1e3", "  not-a-number  "],
)
def test_parse_log_count_rejects_non_integer(arg: str) -> None:
    """Non-integer counts must surface an error string, never crash int()."""
    out = _parse_log_count(arg)
    assert isinstance(out, str)
    assert out.startswith("error:")


# --- _format_status / helpers: empty + ahead/behind/gone edges ----------


def test_format_status_empty_raw_is_clean() -> None:
    """Truly empty porcelain (no branch header) means a clean tree."""
    assert _format_status("") == "[dim]working tree clean[/dim]"


def test_format_status_branch_header_only_is_clean() -> None:
    """A ``##`` header with zero file lines still renders clean, not blank."""
    out = _format_status("## main...origin/main")
    assert "on branch main" in out
    assert "working tree clean" in out


def test_format_branch_line_reports_ahead_behind() -> None:
    """Ahead/behind counts must be humanised so users see divergence."""
    out = _format_branch_line("## main...origin/main [ahead 2, behind 3]")
    assert "on branch main" in out
    assert "2 ahead" in out
    assert "3 behind" in out


def test_format_branch_line_reports_gone_upstream() -> None:
    """A deleted upstream must be flagged, not silently dropped."""
    out = _format_branch_line("## feat...origin/feat [gone]")
    assert "upstream gone" in out


def test_format_branch_line_no_upstream_has_no_suffix() -> None:
    """A branch with no tracking info renders bare, no trailing separator."""
    out = _format_branch_line("## solo")
    assert out == "[bold cyan]on branch solo[/bold cyan]"


def test_format_branch_line_handles_no_space_prefix() -> None:
    """Some porcelain emits ``##branch`` without a space; strip it anyway."""
    out = _format_branch_line("##detached")
    assert "on branch detached" in out


@pytest.mark.parametrize(
    ("line", "style"),
    [
        ("?? new.txt", "dim"),
        (" M edited.txt", "yellow"),
        (" D gone.txt", "red"),
        ("A  added.txt", "green"),
        ("UU conflict.txt", "red"),
        ("XY weird.txt", "dim"),  # unknown code falls back to dim
    ],
)
def test_format_file_line_styles_by_status_code(
    line: str, style: str,
) -> None:
    """Each two-char status code maps to its colour so users scan fast."""
    out = _format_file_line(line)
    assert out == f"[{style}]{line}[/{style}]"


@pytest.mark.parametrize("line", ["", "?", "M"])
def test_format_file_line_passthrough_when_too_short(line: str) -> None:
    """Lines shorter than a status code are returned verbatim, never sliced."""
    assert _format_file_line(line) == line


# --- error paths via the _git seam (no real subprocess) -----------------


@pytest.mark.asyncio
async def test_status_generic_failure_surfaces_stderr(
    agent: AgentSession,
) -> None:
    """A nonzero git that isn't a missing-repo must echo its stderr dimmed."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(1, "", "fatal: index file corrupt"),
    ):
        result = await GitStatusCommand().handle("", agent)
    assert result.handled is True
    assert "index file corrupt" in result.text


@pytest.mark.asyncio
async def test_status_generic_failure_empty_stderr_has_fallback(
    agent: AgentSession,
) -> None:
    """Nonzero exit with empty stderr must still say something, not blank."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(1, "", ""),
    ):
        result = await GitStatusCommand().handle("", agent)
    assert "git status failed" in result.text


@pytest.mark.asyncio
async def test_diff_not_a_repo_returns_error(agent: AgentSession) -> None:
    """/diff outside a repo must produce the same friendly not-a-repo line."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(128, "", "fatal: not a git repository (or any parent)"),
    ):
        result = await GitDiffCommand().handle("", agent)
    assert result.handled is True
    assert "not a git repository" in result.text


@pytest.mark.asyncio
async def test_diff_generic_failure_empty_stderr_has_fallback(
    agent: AgentSession,
) -> None:
    """/diff nonzero with no stderr falls back to a stable error label."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(1, "", ""),
    ):
        result = await GitDiffCommand().handle("", agent)
    assert "git diff failed" in result.text


@pytest.mark.asyncio
async def test_diff_not_installed_returns_friendly_error(
    agent: AgentSession,
) -> None:
    """/diff must catch a missing git binary, not bubble a raw OSError."""
    with patch(
        "aura.application.commands.git._git",
        new=_raise_git(_GitNotInstalledError("git: not found")),
    ):
        result = await GitDiffCommand().handle("", agent)
    assert "git CLI not installed" in result.text


@pytest.mark.asyncio
async def test_diff_timeout_returns_friendly_error(
    agent: AgentSession,
) -> None:
    """/diff timeout must name the command so the user knows what stalled."""
    with patch(
        "aura.application.commands.git._git",
        new=_raise_git(_GitTimeoutError("git timed out after 5s")),
    ):
        result = await GitDiffCommand().handle("", agent)
    assert "/diff timed out" in result.text


@pytest.mark.asyncio
async def test_log_not_a_repo_returns_error(agent: AgentSession) -> None:
    """/log outside a repo must produce the friendly not-a-repo line."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(128, "", "fatal: not a git repository"),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "not a git repository" in result.text


@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: your current branch 'main' does not have any commits yet",
        "fatal: bad default revision 'HEAD'",
        "fatal: ambiguous argument 'HEAD': unknown revision",
    ],
)
@pytest.mark.asyncio
async def test_log_empty_history_variants_say_no_commits(
    stderr: str, agent: AgentSession,
) -> None:
    """All of git's empty-history stderr phrasings map to one calm message."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(128, "", stderr),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "no commits yet" in result.text


@pytest.mark.asyncio
async def test_log_generic_failure_empty_stderr_has_fallback(
    agent: AgentSession,
) -> None:
    """/log nonzero with unrecognised empty stderr falls back to a label."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(1, "", ""),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "git log failed" in result.text


@pytest.mark.asyncio
async def test_log_generic_failure_surfaces_stderr(
    agent: AgentSession,
) -> None:
    """An unexpected /log failure echoes git's own stderr, dimmed."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(1, "", "error: pathspec 'nope' did not match"),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "pathspec 'nope'" in result.text


@pytest.mark.asyncio
async def test_log_not_installed_returns_friendly_error(
    agent: AgentSession,
) -> None:
    """/log must catch a missing git binary like the other commands do."""
    with patch(
        "aura.application.commands.git._git",
        new=_raise_git(_GitNotInstalledError("git: not found")),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "git CLI not installed" in result.text


@pytest.mark.asyncio
async def test_log_timeout_returns_friendly_error(
    agent: AgentSession,
) -> None:
    """/log timeout must name itself so the user knows which command hung."""
    with patch(
        "aura.application.commands.git._git",
        new=_raise_git(_GitTimeoutError("git timed out after 5s")),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "/log timed out" in result.text


@pytest.mark.asyncio
async def test_log_empty_stdout_says_no_commits(
    agent: AgentSession,
) -> None:
    """Exit 0 but whitespace-only stdout still means an empty history."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(0, "   \n", ""),
    ):
        result = await GitLogCommand().handle("", agent)
    assert "no commits yet" in result.text


@pytest.mark.asyncio
async def test_log_appends_trailing_newline_when_missing(
    agent: AgentSession,
) -> None:
    """A final line without its own newline must still be terminated once."""
    console = _capture_buf()
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(0, "abc123 only commit", ""),
    ):
        result = await GitLogCommand(writer=console.write).handle("", agent)
    out = console.getvalue()
    assert out == "abc123 only commit\n"
    assert result.kind == "view"
    assert result.text == ""


@pytest.mark.asyncio
async def test_log_preserves_existing_trailing_newline(
    agent: AgentSession,
) -> None:
    """Output already newline-terminated must not gain a second blank line."""
    console = _capture_buf()
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(0, "abc123 only\n", ""),
    ):
        await GitLogCommand(writer=console.write).handle("", agent)
    assert console.getvalue() == "abc123 only\n"


@pytest.mark.asyncio
async def test_diff_default_writer_falls_back_to_stdout(
    agent: AgentSession,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no injected writer, the diff must reach the real stdout seam."""
    with patch(
        "aura.application.commands.git._git",
        new=_stub_git(0, "diff --git a/x b/x\n+hello\n", ""),
    ):
        result = await GitDiffCommand().handle("--full", agent)
    captured = capsys.readouterr()
    assert "hello" in captured.out
    assert result.kind == "view"
    assert result.text == ""
