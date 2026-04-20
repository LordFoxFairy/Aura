"""Permission subsystem — rules, decisions, modes, safety, persistence.

Spec: ``docs/specs/2026-04-19-aura-permission.md``.
"""

from aura.core.permissions.mode import DEFAULT_MODE, Mode
from aura.core.permissions.safety import DEFAULT_SAFETY, SafetyPolicy, is_protected

__all__ = [
    "DEFAULT_MODE",
    "DEFAULT_SAFETY",
    "Mode",
    "SafetyPolicy",
    "is_protected",
]
