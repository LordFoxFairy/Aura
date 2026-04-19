"""系统提示组装 —— 身份 + 环境。工具通过 bind_tools 注入；记忆由 Context 处理。"""

from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path


def build_system_prompt(
    *,
    cwd: Path | None = None,
    now: dt.datetime | None = None,
) -> str:
    sections = [
        _identity_section(),
        _environment_section(cwd=cwd or Path.cwd(), now=now),
    ]
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
