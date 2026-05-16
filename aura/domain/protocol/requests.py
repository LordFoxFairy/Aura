"""Canonical Aura protocol request payload contracts."""

from __future__ import annotations

from typing import TypedDict


class HeadlessRequest(TypedDict, total=False):
    kind: str
    text: str
    id: str
    choice: str
    feedback: str


__all__ = ["HeadlessRequest"]
