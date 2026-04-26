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
    auto_memory_dir: Path | None = None,
) -> str:
    sections = [
        _identity_section(),
        _environment_section(
            cwd=cwd or Path.cwd(), now=now, model_spec=model_spec,
        ),
    ]
    if auto_memory_dir is not None:
        sections.append(_auto_memory_section(auto_memory_dir))
    return "\n\n".join(sections)


def _auto_memory_section(memory_dir: Path) -> str:
    """F-03-004 — explain the auto-memory convention to the model.

    The model reads + writes memory files via the existing ``write_file``
    / ``read_file`` tools; this prompt section names the directory and
    conventions so the model knows where to put a recall and what types
    of memory to capture. Mirrors claude-code's auto-memory instructions
    almost verbatim — the file layout (one .md per memory + a top-level
    ``MEMORY.md`` index) is identical so users moving between the two
    have a uniform mental model.
    """
    return (
        "# auto memory\n\n"
        f"You have a persistent, file-based memory system at `{memory_dir}/`. "
        "Write to it directly with the ``write_file`` tool — no need to "
        "create the directory first; ``write_file`` materializes parent "
        "directories on demand.\n\n"
        "Build up memory over time so future conversations have a complete "
        "picture of who the user is, how they want to collaborate, what "
        "behaviors to repeat or avoid, and the project's context.\n\n"
        "## Types of memory\n\n"
        "- **user**: role, goals, responsibilities, knowledge level, "
        "preferences. Tailor your behavior to the user's profile.\n"
        "- **feedback**: explicit corrections + validated approaches the "
        "user has confirmed. Save with a short *Why:* line so edge cases "
        "make sense later.\n"
        "- **project**: ongoing work, deadlines, decisions, incidents that "
        "aren't derivable from the code or git history. Convert relative "
        "dates to absolute (\"Thursday\" → \"2026-04-30\") so memory stays "
        "interpretable after time passes.\n"
        "- **reference**: pointers to where information lives in external "
        "systems (Linear projects, Slack channels, Grafana dashboards).\n\n"
        "## What NOT to save\n\n"
        "- Code patterns / file paths / architecture — derivable from the "
        "current project state.\n"
        "- Git history — ``git log`` is authoritative.\n"
        "- Anything already in CLAUDE.md / AURA.md.\n"
        "- Ephemeral task state — that's what conversation context is for.\n\n"
        "## How to save\n\n"
        f"1. Write the memory to its own file (``{memory_dir}/<topic>.md``) "
        "with this frontmatter:\n\n"
        "   ```\n"
        "   ---\n"
        "   name: {memory name}\n"
        "   description: {one-line hook used to decide relevance later}\n"
        "   type: {user, feedback, project, reference}\n"
        "   ---\n\n"
        "   {memory content}\n"
        "   ```\n\n"
        f"2. Append a one-line pointer to ``{memory_dir}/MEMORY.md``:\n\n"
        "   ```\n"
        "   - [Title](file.md) — one-line hook\n"
        "   ```\n\n"
        f"``MEMORY.md`` is the index — keep entries under ~150 chars each "
        "so it stays scannable. The full memory bodies live in their own "
        "files and are loaded on demand via ``read_file``.\n\n"
        "## When to access\n\n"
        "Read memory when it's relevant or when the user references "
        "prior-conversation context. Re-check that a memory is still "
        "correct (read the current code) before acting on it — memories "
        "can go stale. If a remembered fact conflicts with what you "
        "observe now, trust observation and update or remove the stale "
        "memory.\n\n"
        "## Saving on user request\n\n"
        "If the user asks you to remember something, save it immediately. "
        "If they ask you to forget something, find and remove the entry."
    )


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
