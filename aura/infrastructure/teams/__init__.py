"""Teammate backends — strategies for running a teammate's lifecycle."""

from aura.infrastructure.teams.detection import (
    is_inside_tmux,
    pane_backend_available,
    tmux_available,
)
from aura.infrastructure.teams.in_process import InProcessBackend, InProcessHandle
from aura.infrastructure.teams.pane import PaneBackend, PaneHandle
from aura.infrastructure.teams.registry import BackendUnavailable, get_backend
from aura.infrastructure.teams.types import BackendHandle, TeammateBackend

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
