"""Value types shared out of context assembly — leaf, no langchain/loader deps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadRecord:
    mtime: float
    size: int
    partial: bool = False
