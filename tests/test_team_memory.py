"""Team memory: shared notes path + secret-scrubbing redaction.

Covers :mod:`aura.core.teams.memory`. The redactor is load-bearing —
a regression that lets an AWS key leak across the team mailbox is a
real-world incident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aura.core.teams.memory import (
    REDACTION_MARKER,
    append_team_note,
    redact_secrets,
    team_memory_path,
    team_memory_prompt,
)

# ---------------------------------------------------------------------------
# Paths + prompt
# ---------------------------------------------------------------------------


def test_team_memory_path_lives_under_aura_root() -> None:
    p = team_memory_path("alpha")
    # Path is under ``~/.aura/teams/<id>/memory/`` per spec — checking
    # the last three parts keeps the assertion stable across home dirs.
    parts = list(p.parts)
    assert parts[-4:] == [".aura", "teams", "alpha", "memory"]


def test_team_memory_path_does_not_create_directory(tmp_path: Path) -> None:
    # Pure path-resolution — must not touch the filesystem. The autouse
    # ``HOME=tmp`` fixture (tests/conftest.py) means our resolved path
    # lives under the test's isolated home, so we can assert it stays
    # empty until a writer explicitly creates it.
    p = team_memory_path("never-touched")
    assert not p.exists()


def test_team_memory_prompt_mentions_team_and_notes_path() -> None:
    prompt = team_memory_prompt("team-x")
    assert "team-x" in prompt
    assert "notes.md" in prompt
    # The prompt must warn against pasting secrets — the redactor is
    # the safety net, but cooperation from the model is cheaper.
    assert "secret" in prompt.lower() or "credential" in prompt.lower()


# ---------------------------------------------------------------------------
# redact_secrets coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        # AWS access key id — classic shape (alnum-23467 alphabet per AWS).
        "secret: AKIAIOSFODNN7EXAMPLE",
        "ASIA234567ABCDEFGHIJ",
    ],
)
def test_redact_covers_aws_access_keys(raw: str) -> None:
    out = redact_secrets(raw)
    # Original key text must be gone — the prefix substring may survive
    # inside ``[REDACTED]``-adjacent text, so assert against the full
    # token instead.
    assert raw.split(": ")[-1].split()[0] not in out
    assert REDACTION_MARKER in out


def test_redact_covers_anthropic_keys() -> None:
    raw = (
        "Here is my key: sk-ant-api03-" + "A" * 90 + "AA"
    )
    out = redact_secrets(raw)
    assert "sk-ant-api03" not in out
    assert REDACTION_MARKER in out


def test_redact_covers_openai_keys() -> None:
    raw = "OPENAI_API_KEY=sk-proj-" + "A" * 80
    out = redact_secrets(raw)
    # ``sk-proj-`` payload must be gone; the env-line redaction
    # replaces the value, then the generic alnum pattern double-covers
    # any leftover.
    assert "sk-proj-AAAA" not in out
    assert REDACTION_MARKER in out


def test_redact_covers_generic_long_alnum() -> None:
    # 60-char alnum blob with no recognised prefix — likely a token.
    raw = "stash: " + "x9Y" * 25
    out = redact_secrets(raw)
    assert REDACTION_MARKER in out
    # Original blob should be gone.
    assert raw.split(": ")[1] not in out


def test_redact_covers_env_lines() -> None:
    raw = "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    out = redact_secrets(raw)
    assert "AWS_SECRET_ACCESS_KEY=" in out
    assert "wJalrX" not in out
    assert REDACTION_MARKER in out


def test_redact_leaves_short_innocent_text_alone() -> None:
    raw = "Hello world. This is a normal sentence."
    assert redact_secrets(raw) == raw


def test_redact_is_idempotent() -> None:
    raw = "AKIAIOSFODNN7EXAMPLE"
    once = redact_secrets(raw)
    twice = redact_secrets(once)
    assert once == twice


def test_redact_handles_empty_string() -> None:
    assert redact_secrets("") == ""
    assert redact_secrets("   ") == "   "


# ---------------------------------------------------------------------------
# append_team_note
# ---------------------------------------------------------------------------


def test_append_team_note_writes_redacted_line(tmp_path: Path) -> None:
    # ``HOME=tmp`` fixture isolates the .aura dir; the call resolves
    # under the test's home automatically.
    append_team_note(
        "alpha",
        "alice",
        "Cleared cache. Key was AKIAIOSFODNN7EXAMPLE, leaving notes.",
    )
    note_file = team_memory_path("alpha") / "notes.md"
    text = note_file.read_text(encoding="utf-8")
    # Author is preserved, the AWS key is gone.
    assert "alice" in text
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert REDACTION_MARKER in text


def test_append_team_note_creates_directory(tmp_path: Path) -> None:
    append_team_note("bravo", "bob", "Initial note.")
    assert team_memory_path("bravo").is_dir()


def test_append_team_note_ignores_empty_body(tmp_path: Path) -> None:
    # No file should be created for a blank note.
    append_team_note("ghost", "ghost-author", "   ")
    assert not (team_memory_path("ghost") / "notes.md").exists()


def test_append_team_note_appends_rather_than_truncates(tmp_path: Path) -> None:
    append_team_note("charlie", "alice", "first")
    append_team_note("charlie", "bob", "second")
    text = (team_memory_path("charlie") / "notes.md").read_text(encoding="utf-8")
    assert "first" in text and "second" in text
    # Two lines, in order — append-only contract.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0].index("first") >= 0
    assert lines[1].index("second") >= 0
