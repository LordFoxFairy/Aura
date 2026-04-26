"""Environment detection for the pane backend.

Three predicates, deliberately single-purpose so tests can monkeypatch
them in isolation:

- :func:`is_inside_tmux` — is the current process running under a tmux
  client (i.e. ``$TMUX`` is set)?
- :func:`tmux_available` — is the ``tmux`` binary on ``$PATH``?
- :func:`pane_backend_available` — both of the above; the registry
  uses this as the gate before instantiating
  :class:`~aura.core.teams.backends.pane.PaneBackend`.

We deliberately do NOT cache results: the user may switch sessions
mid-run (rare but possible), and the cost of an env-var lookup +
``shutil.which`` is negligible compared to a tmux IPC round-trip.
"""

from __future__ import annotations

import os
import shutil


def is_inside_tmux() -> bool:
    """Return ``True`` iff ``$TMUX`` is set.

    tmux sets this env var on every child process it spawns; checking
    its presence is the documented way to detect "am I being run
    inside a tmux session?" (man tmux, ENVIRONMENT). A detached tmux
    server has children with ``$TMUX`` unset — that's correct, we
    can't split a pane in a session we're not attached to.
    """
    return bool(os.environ.get("TMUX"))


def tmux_available() -> bool:
    """Return ``True`` iff a ``tmux`` binary is reachable via ``$PATH``."""
    return shutil.which("tmux") is not None


def pane_backend_available() -> bool:
    """Combined gate: pane backend can run iff both predicates hold."""
    return is_inside_tmux() and tmux_available()


__all__ = ["is_inside_tmux", "pane_backend_available", "tmux_available"]
