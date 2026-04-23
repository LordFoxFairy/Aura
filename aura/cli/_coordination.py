"""Cross-module CLI coordination — single-writer state for "get off the screen".

Two coordination points live here:

1. **Spinner pause**: the thinking spinner renders via ``rich.Live`` at
   10 fps. The permission asker wants to render an interactive
   ``prompt_toolkit.Application`` over the same terminal. If both
   render at once, the result is unreadable.

2. **Prompt serialization**: pt's ``Application`` needs exclusive
   ownership of the terminal. With subagents running concurrent tool
   calls on the same event loop, two askers (parent's permission,
   subagent's ``ask_user_question``, etc.) could both try to start a
   pt Application — chaos. ``prompt_mutex`` serializes all
   user-interaction widgets so they render one at a time, in FIFO
   order (asyncio.Lock is queue-fair).

The cleanest cross-module alternative would be a proper event bus,
but that's overkill for two coordination points. API kept minimal on
purpose — anything more elaborate belongs in a real event bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

_pause_spinner: Callable[[], Awaitable[None]] | None = None

# Lazily constructed so the lock is bound to the event loop that first
# uses it (instead of a module-import-time loop that may be stale
# after ``asyncio.run`` exits and returns in another loop).
_prompt_mutex: asyncio.Lock | None = None


def set_spinner_pause_callback(
    callback: Callable[[], Awaitable[None]] | None,
) -> None:
    """Register (or clear) the "pause current spinner" callback.

    Called by the REPL with a bound ``ThinkingSpinner.stop`` method at
    the start of each turn, and ``None`` when no spinner is active.
    """
    global _pause_spinner
    _pause_spinner = callback


async def pause_spinner_if_active() -> None:
    """Invoke the registered pause callback, if any.

    Idempotent — safe to call when no spinner is active (no-op). After
    a successful pause, the callback slot is cleared so a second call
    in the same turn doesn't await a stopped spinner.
    """
    global _pause_spinner
    callback = _pause_spinner
    if callback is not None:
        _pause_spinner = None
        await callback()


def prompt_mutex() -> asyncio.Lock:
    """Return the process-wide prompt serialization lock.

    Every interactive widget (permission asker, ask_user_question
    asker, any future confirm dialog) MUST acquire this before
    entering its ``pt.Application.run_async``. That guarantees at most
    one widget owns the terminal at a time, and concurrent requests
    queue FIFO (asyncio.Lock is queue-fair by default).

    Lazy construction: built on first call so the Lock is bound to
    whatever event loop is running at the time. Each asyncio.run()
    session gets its own lock; tests that tear down and rebuild loops
    don't hit "Lock is bound to a different loop" errors.
    """
    global _prompt_mutex
    if _prompt_mutex is None:
        _prompt_mutex = asyncio.Lock()
    return _prompt_mutex


def _reset_prompt_mutex_for_tests() -> None:
    """Drop the cached lock so the next ``prompt_mutex()`` call builds
    a fresh one on whatever loop the test is using. NEVER call from
    production code."""
    global _prompt_mutex
    _prompt_mutex = None
