"""Team memory: secret-scrubbing redaction.

The redactor is wired into :class:`TeamManager` so every text body that
crosses a member boundary is scrubbed before it can be persisted or
relayed. Conservative — false positives are OK, false negatives are not.
"""

from __future__ import annotations

import re

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # AWS access key id
    re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z2-7]{16}\b"),
    # Anthropic Claude API keys
    re.compile(r"\bsk-ant(?:-admin01|-api)?-[A-Za-z0-9_-]{20,}\b"),
    # OpenAI keys (newer prefixed shapes + classic sk-…)
    re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    # GCP API keys
    re.compile(r"\bAIza[\w-]{35}\b"),
    # GitHub personal access tokens (classic + fine-grained)
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b"),
    # Generic long base64/hex blobs — 40+ alnum/_- is well above prose noise.
    re.compile(r"\b[A-Za-z0-9_-]{40,}\b"),
)

_ENV_LINE_RE = re.compile(
    r"(?im)^\s*"
    r"(?P<key>[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASS(?:WORD)?|CRED|API)[A-Z0-9_]*)"
    r"\s*=\s*"
    r"(?P<value>\S+)",
)

REDACTION_MARKER = "[REDACTED]"


def redact_secrets(text: str) -> str:
    """Replace well-known secret shapes in ``text`` with ``[REDACTED]``.

    Handles AWS / Anthropic / OpenAI / GCP / GitHub credentials, generic
    long alnum/base64 blobs, and ``KEY=VALUE`` lines where the key looks
    credential-ish. Idempotent on already-redacted input.
    """
    if not text or not text.strip():
        return text
    # Env-style lines first so the value isn't eaten by the generic pattern.
    text = _ENV_LINE_RE.sub(
        lambda m: f"{m.group('key')}={REDACTION_MARKER}", text,
    )
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTION_MARKER, text)
    return text
