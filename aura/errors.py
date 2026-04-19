"""Root exception type for Aura."""

from __future__ import annotations


class AuraError(Exception):
    """Base class for expected, user-facing Aura errors.

    Callers that want to uniformly handle "something Aura itself knows how to
    report" should `except AuraError`. Stdlib / third-party exceptions such as
    `asyncio.CancelledError` or pydantic `ValidationError` are intentionally
    not caught here — those need their own handlers.
    """


__all__ = ["AuraError"]
