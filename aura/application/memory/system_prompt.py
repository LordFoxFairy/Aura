"""Identity + `<env>` block for the SystemMessage."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path

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

# Slow / network-mounted repos must not bottleneck startup.
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
    if not model_spec:
        return None
    direct = KNOWLEDGE_CUTOFFS.get(model_spec)
    if direct:
        return direct
    # Strip leading "provider:" tag (e.g. "anthropic:claude-opus-4-7").
    tail = model_spec.split(":", 1)[-1]
    return KNOWLEDGE_CUTOFFS.get(tail)


def _git_status_line(cwd: Path) -> str | None:
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
        # Forms: "## main", "## main...origin/main", "## HEAD (no branch)".
        branch_part = first[3:].split("...", 1)[0].strip()
        if branch_part:
            branch = branch_part
    dirty = any(not ln.startswith("##") for ln in lines)
    state = "dirty" if dirty else "clean"
    return f"git: {branch} ({state})"
