"""Teammate backends — strategies for running a teammate's lifecycle.

Mirrors claude-code's two-backend pattern (``utils/swarm/teamHelpers.ts``
+ ``backends/registry.ts``):

- :class:`InProcessBackend` runs the teammate as an asyncio task in the
  leader's process (default; matches the original Aura behaviour).
- :class:`PaneBackend` spawns the teammate as a subprocess inside a
  tmux pane (visual "switch into team" UX; requires ``$TMUX`` and the
  ``tmux`` binary on PATH).

The :func:`get_backend` registry returns the singleton instance for a
given :data:`~aura.domain.team.BackendType` literal. Pane backend
construction raises :class:`BackendUnavailable` early when the
environment can't support it, so the manager fails fast on a misrouted
``add_member(backend_type="pane")`` rather than half-spawning a member.
"""

from aura.infrastructure.teams_backends.detection import (
    is_inside_tmux,
    pane_backend_available,
    tmux_available,
)
from aura.infrastructure.teams_backends.in_process import InProcessBackend, InProcessHandle
from aura.infrastructure.teams_backends.pane import PaneBackend, PaneHandle
from aura.infrastructure.teams_backends.registry import BackendUnavailable, get_backend
from aura.infrastructure.teams_backends.types import BackendHandle, TeammateBackend

__all__ = [
    "BackendHandle",
    "BackendUnavailable",
    "InProcessBackend",
    "InProcessHandle",
    "PaneBackend",
    "PaneHandle",
    "TeammateBackend",
    "get_backend",
    "is_inside_tmux",
    "pane_backend_available",
    "tmux_available",
]
