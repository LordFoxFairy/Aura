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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aura.cli.commands import build_default_registry
from aura.core.agent import Agent
from aura.core.commands.git_commands import (
    GitDiffCommand,
    GitLogCommand,
    GitStatusCommand,
    _git,
    _GitTimeoutError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
def agent() -> Agent:
    """MagicMock Agent — none of our commands touch agent state."""
    return MagicMock(spec=Agent)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_git_commands_registered_in_default_registry() -> None:
    r = build_default_registry()
    names = {c.name for c in r.list()}
    assert "/status" in names
    assert "/diff" in names
    assert "/log" in names


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_clean_tree_prints_working_tree_clean(
    repo: Path, agent: Agent,
) -> None:
    _commit(repo, "a.txt", "hello\n", "initial")

    result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "working tree clean" in result.text
    assert "on branch main" in result.text


@pytest.mark.asyncio
async def test_status_with_modified_files_shows_them(
    repo: Path, agent: Agent,
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: Agent,
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


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_default_shows_stat_summary(
    repo: Path, agent: Agent,
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
    repo: Path, agent: Agent,
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
    repo: Path, agent: Agent,
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
async def test_diff_rejects_unknown_flag(agent: Agent) -> None:
    result = await GitDiffCommand().handle("--bogus", agent)
    assert result.handled is True
    assert result.text.startswith("error:")
    assert "--bogus" in result.text


@pytest.mark.asyncio
async def test_diff_truncates_at_500_lines(
    repo: Path, agent: Agent,
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


# ---------------------------------------------------------------------------
# /log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_empty_repo_prints_no_commits_yet(
    repo: Path, agent: Agent,
) -> None:
    result = await GitLogCommand().handle("", agent)

    assert result.handled is True
    assert "no commits yet" in result.text


@pytest.mark.asyncio
async def test_log_with_commits_shows_them(
    repo: Path, agent: Agent,
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
    repo: Path, agent: Agent,
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
    repo: Path, agent: Agent,
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
        "aura.core.commands.git_commands._git", side_effect=fake_git,
    ):
        await GitLogCommand().handle("200", agent)

    # Verify -100 made it in (not -200).
    assert "-100" in captured["args"]
    assert "-200" not in captured["args"]


@pytest.mark.asyncio
async def test_log_rejects_non_integer_arg(agent: Agent) -> None:
    result = await GitLogCommand().handle("abc", agent)
    assert result.handled is True
    assert result.text.startswith("error:")


# ---------------------------------------------------------------------------
# Timeout + git-not-installed (mocked — real timeout would add 5 s)
# ---------------------------------------------------------------------------


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: Agent,
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
        patch("aura.core.commands.git_commands._DEFAULT_TIMEOUT_S", 0.05),
    ):
        result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "timed out" in result.text


@pytest.mark.asyncio
async def test_git_not_installed_returns_friendly_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: Agent,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def fake_exec(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError(2, "No such file", "git")

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await GitStatusCommand().handle("", agent)

    assert result.handled is True
    assert "git CLI not installed" in result.text
