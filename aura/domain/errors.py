"""Root exception type for Aura."""

from __future__ import annotations


class AuraError(Exception):
    """Base class for expected, user-facing Aura errors."""


__all__ = ["AuraError"]
