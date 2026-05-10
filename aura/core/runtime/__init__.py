"""Per-Agent runtime collaborators (Phase 1+ extractions).

Houses the lifecycle / persistence / IO sidecars that started life inside
the :class:`aura.core.agent.Agent` god object. Each module owns one
narrow concern so :class:`Agent` can shrink toward its eventual
≤600-line target without juggling unrelated responsibilities inline.

Phase 1 ships :class:`SessionRuntime` (storage + session_id + history
load/save + clear/resume/aclose lifecycle). Phase 2 will land the MCP
runtime; Phase 6 will land the subagent runtime.
"""
from __future__ import annotations

from aura.core.runtime.session import SessionRuntime

__all__ = ["SessionRuntime"]
