"""Compaction result value types — leaf, no orchestration deps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CompactSource = Literal["manual", "auto", "reactive"]


@dataclass(frozen=True)
class CompactResult:
    before_tokens: int
    after_tokens: int
    source: CompactSource
