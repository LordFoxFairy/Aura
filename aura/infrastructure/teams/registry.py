"""Backend registry — ``get_backend(backend_type)`` with env-gated pane construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.infrastructure.teams.detection import pane_backend_available
from aura.infrastructure.teams.in_process import InProcessBackend
from aura.infrastructure.teams.pane import PaneBackend, PaneBackendError

if TYPE_CHECKING:
    from aura.domain.team import BackendType
    from aura.infrastructure.teams.types import TeammateBackend


class BackendUnavailable(RuntimeError):
    pass


# Stateless backends — a race during first lookup just discards the loser's init.
_in_process_singleton: InProcessBackend | None = None
_pane_singleton: PaneBackend | None = None


def get_backend(backend_type: BackendType) -> TeammateBackend:
    """Return the singleton backend; ``"pane"`` raises when ``$TMUX`` or the binary is missing."""
    global _in_process_singleton, _pane_singleton  # noqa: PLW0603  # lazy init needs module-scope rebind
    if backend_type == "in_process":
        if _in_process_singleton is None:
            _in_process_singleton = InProcessBackend()
        return _in_process_singleton
    if backend_type == "pane":
        if not pane_backend_available():
            raise BackendUnavailable(
                "pane backend unavailable: requires running inside a tmux "
                "session ($TMUX set) AND tmux on PATH",
            )
        if _pane_singleton is None:
            _pane_singleton = PaneBackend()
        return _pane_singleton
    raise BackendUnavailable(f"unknown backend_type: {backend_type!r}")


__all__ = ["BackendUnavailable", "PaneBackendError", "get_backend"]
