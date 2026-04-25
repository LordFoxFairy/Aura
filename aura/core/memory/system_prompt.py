"""系统提示组装 —— 身份 + 环境。工具通过 bind_tools 注入；记忆由 Context 处理。

Round 3A-extended ``<env>`` block: working dir + platform + Python +
date and (when discoverable) git branch and dirty/clean state. The
git probe is best-effort: a 1-second wall-clock timeout, no shell
spawn beyond ``git status --porcelain=v1 -b``, fail-open silently
when ``git`` is missing or the directory isn't a repo. Knowledge-
cutoff hints are indexed by model spec via :data:`KNOWLEDGE_CUTOFFS`
— callers pass ``model_spec=`` to surface the right line; unknown
specs simply omit the line.
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path

#: Knowledge-cutoff lookup keyed by model spec / family. Lookups try
#: an exact match first, then progressively shorter prefixes — so
#: ``anthropic:claude-opus-4-7`` resolves via the ``claude-opus-4-7``
#: row even when the provider tag is included. Values are ``YYYY-MM``
#: strings; absence of an entry means "don't render the line".
KNOWLEDGE_CUTOFFS: dict[str, str] = {
    "claude-opus-4-7": "2026-01",
    "claude-opus-4-6": "2025-09",
    "claude-sonnet-4-7": "2026-01",
    "claude-sonnet-4-6": "2025-07",
    "claude-haiku-4-7": "2026-01",
    "gpt-4o": "2024-10",
    "gpt-4o-mini": "2024-07",
    "gpt-5": "2025-09",
}

#: Wall-clock cap for the git status probe. The block must not
#: bottleneck startup on a slow / network-mounted repo, so anything
#: that runs >1s is treated as "unavailable" and the line is omitted.
_GIT_TIMEOUT_SECONDS: float = 1.0


def build_system_prompt(
    *,
    cwd: Path | None = None,
    now: dt.datetime | None = None,
    model_spec: str | None = None,
) -> str:
    sections = [
        _identity_section(),
        _environment_section(
            cwd=cwd or Path.cwd(), now=now, model_spec=model_spec,
        ),
    ]
    return "\n\n".join(sections)


def _identity_section() -> str:
    return (
        "You are Aura, a general-purpose Python agent with an explicit async loop. "
        "You own tool dispatch; the user sees streaming events (assistant text, tool "
        "calls, final). Be concise, honest about failures, and prefer tool action over "
        "narration. When you don't know, say so."
    )


def _environment_section(
    *,
    cwd: Path,
    now: dt.datetime | None,
    model_spec: str | None,
) -> str:
    current = now or dt.datetime.now().astimezone()
    lines: list[str] = [
        "<env>",
        f"date: {current.strftime('%Y-%m-%d %Z')}",
        f"cwd: {cwd}",
        f"platform: {platform.system()} {platform.release()}",
        f"python: {platform.python_version()}",
    ]
    if model_spec:
        lines.append(f"model: {model_spec}")
        cutoff = _lookup_cutoff(model_spec)
        if cutoff:
            lines.append(f"knowledge_cutoff: {cutoff}")
    git_line = _git_status_line(cwd)
    if git_line:
        lines.append(git_line)
    lines.append("</env>")
    return "\n".join(lines)


def _lookup_cutoff(model_spec: str) -> str | None:
    """Resolve ``model_spec`` to a knowledge-cutoff hint or ``None``.

    Tries an exact match, then strips a leading provider tag
    (``"anthropic:"``, ``"openai:"``, …), then drops trailing version
    suffixes one-by-one. Mirrors the lookup pattern claude-code uses
    for its own banner — robust against ``provider:family-variant``
    naming without exploding the table.
    """
    if not model_spec:
        return None
    direct = KNOWLEDGE_CUTOFFS.get(model_spec)
    if direct:
        return direct
    tail = model_spec.split(":", 1)[-1]
    return KNOWLEDGE_CUTOFFS.get(tail)


def _git_status_line(cwd: Path) -> str | None:
    """Probe ``git status --porcelain=v1 -b`` and render a single line.

    Returns a string like ``"git: main (clean)"`` or ``"git: main (dirty)"``
    when the probe succeeds; ``None`` otherwise (no git, not a repo,
    timeout, OS error). Stderr is squashed — this is best-effort
    metadata, not a primary tool path.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-b"],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    if not lines:
        return None
    branch = "?"
    first = lines[0]
    if first.startswith("## "):
        # Possible forms: "## main", "## main...origin/main",
        # "## HEAD (no branch)".
        branch_part = first[3:].split("...", 1)[0].strip()
        if branch_part:
            branch = branch_part
    dirty = any(not ln.startswith("##") for ln in lines)
    state = "dirty" if dirty else "clean"
    return f"git: {branch} ({state})"
