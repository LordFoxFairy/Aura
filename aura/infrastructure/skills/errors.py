"""Shared error formatting for skill-invocation paths.

Both the slash surface and the tool surface enforce the same
"declared arguments must all be provided" contract; the text must match
byte-for-byte across paths so users / models recognise it on either.
"""

from __future__ import annotations


def format_missing_args_error(
    name: str, declared: tuple[str, ...], provided: int,
) -> str:
    """Canonical missing-required-args message."""
    return (
        f"skill {name!r} requires arguments {list(declared)}; "
        f"missing: {list(declared[provided:])}"
    )
