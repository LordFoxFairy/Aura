"""Compatibility façade for the capabilities-owned command registry."""

from aura.capabilities.commands.registry import (
    CommandRegistry,
    build_default_registry,
    dispatch,
)

__all__ = ["CommandRegistry", "build_default_registry", "dispatch"]
