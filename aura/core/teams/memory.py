"""Team memory: shared notes + secret-scrubbing redaction.

Translates claude-code's ``memdir/teamMemPaths.ts`` +
``memdir/teamMemPrompts.ts`` + ``services/teamMemorySync/
teamMemSecretGuard.ts`` into the simplest Python surface that covers
Aura's current needs.

Public surface:

- :func:`team_memory_path` — resolve the on-disk root for a team's
  shared memory (``~/.aura/teams/<team_id>/memory/``).
- :func:`team_memory_prompt` — system-prompt fragment all team
  members get so the LLM knows shared notes are available.
- :func:`redact_secrets` — regex-scrub well-known secret shapes
  (AWS / OpenAI / Anthropic / env-style assignments / long
  base64-looking tokens). Conservative: false positives are OK,
  false negatives are not.
- :func:`append_team_note` — append a line to the team's shared
  ``notes.md``, redacting secrets before write.

The redactor is the load-bearing piece: it's wired into the team
manager so every text body that crosses a member boundary is scrubbed
before it can be persisted or relayed. Operators relying on the team
to leak their AWS keys would be unhappy.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

# Filename for the append-only shared notes log. Markdown so a human
# editor can open it directly; one entry per line keeps the file
# tail-friendly without a full parser.
_NOTES_FILENAME = "notes.md"


def _team_memory_root() -> Path:
    """Resolve the team-memory root at CALL TIME.

    Re-evaluating ``Path.home()`` on every call (instead of caching at
    import time) means a test that swaps ``HOME`` via monkeypatch
    actually sees its isolated home — without this, the first test
    to import the module would pin the path for every subsequent
    test in the same process.
    """
    return Path.home() / ".aura" / "teams"


def team_memory_path(team_id: str) -> Path:
    """Return the directory holding a team's shared memory artefacts.

    Equivalent to claude-code's ``getTeamMemPath``. The directory is
    NOT created here — callers that need to write should call
    :meth:`Path.mkdir` themselves so a read-only consumer doesn't
    accidentally materialise an empty tree.
    """
    return _team_memory_root() / team_id / "memory"


def team_memory_prompt(team_id: str) -> str:
    """Return the system-prompt fragment all team members receive.

    Inlined verbatim into the team agent's system prompt so the LLM
    knows the shared note file exists + understands the protocol
    (append, don't rewrite; refer to teammates by name).
    """
    notes_path = team_memory_path(team_id) / _NOTES_FILENAME
    return (
        f"\n\n# Team memory\n"
        f"You are a member of team {team_id!r}. The team shares a notes file at "
        f"`{notes_path}`. When you make a decision or learn a fact that other "
        f"teammates should know, append a short note (one line per insight). "
        f"Read existing notes before starting work so you do not duplicate "
        f"effort. Never include secrets / API keys / credentials in notes — "
        f"the file is shared with every member.\n"
    )


# --- Secret-scrubbing rules -------------------------------------------------

# Patterns chosen for HIGH confidence — false positives are OK
# (operators occasionally see an over-eager ``[REDACTED]`` and shrug),
# false negatives are not. Each pattern is anchored on a distinctive
# prefix or shape so generic prose doesn't trigger by accident.
#
# Ordering matters slightly: more specific patterns are tried first so
# the generic catch-all doesn't shadow a labelled match.

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # AWS access key id — the AKIA / ASIA / etc. prefix is unique to
    # AWS and the 20-char alnum body is well-defined.
    re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16}\b"),
    # Anthropic Claude API keys — ``sk-ant-…``. Tolerate both the
    # full 100+-char production keys and the shorter admin variants.
    re.compile(r"\bsk-ant(?:-admin01|-api)?-[A-Za-z0-9_-]{20,}\b"),
    # OpenAI keys — old ``sk-…`` shape AND the newer ``sk-proj-…`` /
    # ``sk-svcacct-…`` shapes. We keep both alive because rotating
    # users have a mix on hand for months.
    re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    # GCP API keys — ``AIza…`` prefix is unique to Google.
    re.compile(r"\bAIza[\w-]{35}\b"),
    # GitHub personal access tokens (classic + fine-grained).
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
    # Generic long base64 / hex blobs. 40+ chars of alnum + ``-_`` is
    # well above the random-prose noise floor — false positives on
    # long identifiers are tolerable; false negatives on accidentally
    # pasted JWTs / signing keys are not.
    re.compile(r"\b[A-Za-z0-9_-]{40,}\b"),
)

# Env-style ``KEY=VALUE`` assignments where the key looks
# credential-ish. Single line per match — we don't try to walk
# multi-line .env blocks; the per-line redaction is enough.
_ENV_LINE_RE = re.compile(
    r"(?im)^\s*"
    r"(?P<key>[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASS(?:WORD)?|CRED|API)[A-Z0-9_]*)"
    r"\s*=\s*"
    r"(?P<value>\S+)",
)

# Replacement marker. Visible enough that an operator scanning a
# scrubbed log immediately sees the redaction; structured enough that
# a downstream consumer can pattern-match on it.
REDACTION_MARKER = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace well-known secret shapes in ``text`` with ``[REDACTED]``.

    Conservative — over-redacts rather than leaking. Handles:

    1. AWS access key ids (``AKIA…``)
    2. Anthropic API keys (``sk-ant-…``)
    3. OpenAI keys (``sk-…`` and the newer prefixed variants)
    4. GCP API keys (``AIza…``)
    5. GitHub PATs (``ghp_…`` / ``github_pat_…``)
    6. Generic long base64/hex/alnum strings (40+ chars)
    7. ``KEY=VALUE`` lines where ``KEY`` looks credential-ish
       (anything containing ``KEY`` / ``SECRET`` / ``TOKEN`` /
       ``PASSWORD`` / ``CRED`` / ``API``)

    Idempotent — running redact on already-redacted text is a no-op.
    Empty / whitespace-only input passes through unchanged.
    """
    if not text or not text.strip():
        return text
    # Env-style lines first so the captured value isn't already eaten
    # by the generic long-string pattern. We replace just the value
    # portion so the key stays visible — an operator scanning the log
    # still sees ``AWS_ACCESS_KEY_ID=[REDACTED]`` and can grep.
    text = _ENV_LINE_RE.sub(
        lambda m: f"{m.group('key')}={REDACTION_MARKER}",
        text,
    )
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTION_MARKER, text)
    return text


def append_team_note(team_id: str, author: str, note: str) -> None:
    """Append a redacted note to a team's shared ``notes.md``.

    Format::

        <iso-timestamp> <author>: <note-with-secrets-redacted>

    The directory is created on demand (mkdir parents=True). Empty /
    whitespace-only notes are silently ignored — appending blank
    lines would clutter the file without adding signal.

    Redaction runs on the note body BEFORE write so even an authorised
    member can't accidentally commit an API key into the shared log.
    """
    body = note.strip()
    if not body:
        return
    scrubbed_author = redact_secrets(author).strip() or "anonymous"
    scrubbed_body = redact_secrets(body)
    target = team_memory_path(team_id) / _NOTES_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{timestamp} {scrubbed_author}: {scrubbed_body}\n"
    with target.open("a", encoding="utf-8") as f:
        f.write(line)


__all__ = [
    "REDACTION_MARKER",
    "append_team_note",
    "redact_secrets",
    "team_memory_path",
    "team_memory_prompt",
]
