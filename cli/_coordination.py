"""Cross-module CLI coordination — single-writer state for prompt widgets."""

from __future__ import annotations

import asyncio

# Lazy so the lock binds to the event loop that first uses it (re-running
# asyncio.run produces a fresh loop).
_prompt_mutex: asyncio.Lock | None = None


async def pause_spinner_if_active() -> None:
    # Kept as a cheap awaitable for callers that still invoke it.
    return None


def prompt_mutex() -> asyncio.Lock:
    global _prompt_mutex
    if _prompt_mutex is None:
        _prompt_mutex = asyncio.Lock()
    return _prompt_mutex


def _reset_prompt_mutex_for_tests() -> None:
    global _prompt_mutex
    _prompt_mutex = None
