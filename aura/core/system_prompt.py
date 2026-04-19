"""系统提示组装 —— 身份/环境/工具指引/项目记忆的组合。"""

from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

from aura.core.registry import ToolRegistry

_AURA_MD_FILENAME = "AURA.md"
_AURA_MD_MAX_BYTES = 10 * 1024  # 10 KB cap — 超限截断，防巨型 README 吞 context
_AURA_MD_MAX_WALK = 10  # 向上最多走 10 级父目录


def build_system_prompt(
    *,
    registry: ToolRegistry,
    cwd: Path | None = None,
    now: dt.datetime | None = None,
) -> str:
    sections: list[str] = []
    sections.append(_identity_section())
    sections.append(_environment_section(cwd=cwd or Path.cwd(), now=now))
    sections.append(_tools_section(registry))
    memory = _find_and_load_aura_md(cwd or Path.cwd())
    if memory:
        sections.append(memory)
    return "\n\n".join(sections)


def _identity_section() -> str:
    return (
        "You are Aura, a general-purpose Python agent with an explicit async loop. "
        "You own tool dispatch; the user sees streaming events (assistant text, tool "
        "calls, final). Be concise, honest about failures, and prefer tool action over "
        "narration. When you don't know, say so."
    )


def _environment_section(*, cwd: Path, now: dt.datetime | None) -> str:
    current = now or dt.datetime.now().astimezone()
    return (
        f"<env>\n"
        f"date: {current.strftime('%Y-%m-%d %Z')}\n"
        f"cwd: {cwd}\n"
        f"platform: {platform.system()} {platform.release()}\n"
        f"python: {platform.python_version()}\n"
        f"</env>"
    )


def _tools_section(registry: ToolRegistry) -> str:
    if len(registry) == 0:
        return "<tools>none enabled</tools>"
    lines = ["<tools>"]
    for tool in registry:
        meta = tool.metadata or {}
        flags = []
        if meta.get("is_read_only", False):
            flags.append("read-only")
        if meta.get("is_destructive", False):
            flags.append("destructive")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"- {tool.name}{flag_str}: {tool.description}")
    lines.append("</tools>")
    return "\n".join(lines)


def _find_and_load_aura_md(cwd: Path) -> str | None:
    cwd = cwd.resolve()
    for ancestor in [cwd, *cwd.parents][: _AURA_MD_MAX_WALK + 1]:
        candidate = ancestor / _AURA_MD_FILENAME
        if not candidate.is_file():
            continue
        try:
            data = candidate.read_bytes()[: _AURA_MD_MAX_BYTES + 1]
        except OSError:
            return None
        truncated = len(data) > _AURA_MD_MAX_BYTES
        text = data[:_AURA_MD_MAX_BYTES].decode("utf-8", errors="replace")
        header = f"<project_memory source={candidate}>"
        footer = "</project_memory>"
        if truncated:
            text += "\n… (truncated)"
        return f"{header}\n{text}\n{footer}"
    return None
