"""Provider context-overflow detection — pure exception classifier."""

from __future__ import annotations

# Match on stringified message so provider SDK types stay out of imports.
_CONTEXT_OVERFLOW_PHRASES: tuple[str, ...] = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "prompt is too long",
    "prompt exceeds max length",
    "input too long",
    "exceeds max length",
    "request payload size exceeds",
    "too many tokens",
    "max_tokens exceeded",
)

# Structured codes survive SDK message localisation; phrase match would miss.
_CONTEXT_OVERFLOW_CODES: tuple[str, ...] = (
    "1261",
)


def is_context_overflow(exc: BaseException) -> bool:
    """True iff ``exc`` matches a known provider context-overflow signature."""
    msg = str(exc).lower()
    if any(phrase in msg for phrase in _CONTEXT_OVERFLOW_PHRASES):
        return True
    for code in _CONTEXT_OVERFLOW_CODES:
        if f"'code': '{code}'" in msg or f'"code": "{code}"' in msg:
            return True
    return False
