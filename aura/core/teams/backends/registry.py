"""Backend registry — singleton lookup keyed by :data:`BackendType`.

Mirrors claude-code's ``backends/registry.ts`` pattern: a single
``get_backend(backend_type)`` entry point that returns the matching
strategy, with an environment guard for the pane backend so a misrouted
``add_member(backend_type="pane")`` outside tmux fails fast with a clear
message instead of half-spawning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aura.core.teams.backends.detection import pane_backend_available
from aura.core.teams.backends.in_process import InProcessBackend
from aura.core.teams.backends.pane import PaneBackend, PaneBackendError

if TYPE_CHECKING:
    from aura.core.teams.backends.types import TeammateBackend
    from aura.core.teams.types import BackendType


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend can't run in this environment.

    The manager catches this and surfaces it as a TeamError, so the
    caller sees a clear "you asked for the pane backend, but you're
    not inside tmux" rather than a stray RuntimeError.
    """


# Module-level singletons — lazy-initialized on first lookup. Stateless
# backends, so concurrent first-lookups racing is harmless (the second
# init is just discarded).
_in_process_singleton: InProcessBackend | None = None
_pane_singleton: PaneBackend | None = None


def get_backend(backend_type: BackendType) -> TeammateBackend:
    """Return the singleton backend for ``backend_type``.

    Raises :class:`BackendUnavailable` for ``"pane"`` when the
    environment lacks tmux (no ``$TMUX`` or no binary on PATH). The
    in-process backend is always available.
    """
    global _in_process_singleton, _pane_singleton  # noqa: PLW0603
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
    # Future backend types (e.g. iterm2 split, ssh) plug in here. Until
    # then a Literal-violating value is a programmer error.
    raise BackendUnavailable(f"unknown backend_type: {backend_type!r}")


def _reset_for_tests() -> None:
    """Reset the singletons. Test-only — do not call from production code.

    Tests that monkeypatch ``pane_backend_available`` need a fresh
    instance afterward; this helper short-circuits the cache without
    exposing the module globals.
    """
    global _in_process_singleton, _pane_singleton  # noqa: PLW0603
    _in_process_singleton = None
    _pane_singleton = None


__all__ = ["BackendUnavailable", "PaneBackendError", "get_backend"]
