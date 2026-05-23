"""Cross-module CLI coordination — single-writer state for prompt widgets."""

from __future__ import annotations

import asyncio

# Lazy so the lock binds to the event loop that first uses it.
_prompt_mutex: asyncio.Lock | None = None


async def pause_spinner_if_active() -> None:
    return None


def prompt_mutex() -> asyncio.Lock:
    global _prompt_mutex
    if _prompt_mutex is None:
        _prompt_mutex = asyncio.Lock()
    return _prompt_mutex
