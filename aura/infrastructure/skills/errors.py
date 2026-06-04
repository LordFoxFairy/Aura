"""Shared missing-args error text so slash and tool surfaces match byte-for-byte."""

from __future__ import annotations


def format_missing_args_error(
    name: str,
    declared: tuple[str, ...],
    provided: int,
) -> str:
    """Canonical missing-required-args message."""
    return (
        f"skill {name!r} requires arguments {list(declared)}; missing: {list(declared[provided:])}"
    )
