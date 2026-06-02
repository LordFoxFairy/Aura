"""Permission Outcome tagged union: Allow / Block / Ask / Replace."""

from __future__ import annotations

from dataclasses import dataclass

from aura.domain.permission.decision import Decision
from aura.domain.tool import ToolResult


@dataclass(frozen=True)
class Allow:
    decision: Decision

    def __post_init__(self) -> None:
        if not self.decision.allow:
            raise ValueError(
                "Allow requires a Decision with allow=True; "
                f"got reason={self.decision.reason!r}, allow=False"
            )


@dataclass(frozen=True)
class Block:
    decision: Decision

    def __post_init__(self) -> None:
        if self.decision.allow:
            raise ValueError(
                "Block requires a Decision with allow=False; "
                f"got reason={self.decision.reason!r}, allow=True"
            )


@dataclass(frozen=True)
class Ask:
    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError("Ask requires a non-empty reason")


@dataclass(frozen=True)
class Replace:
    result: ToolResult
    decision: Decision

    def __post_init__(self) -> None:
        # Guards off-type construction that bypasses the non-Optional annotation.
        if self.result is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("Replace requires a non-None ToolResult")
        if self.decision.allow:
            raise ValueError(
                "Replace requires a Decision with allow=False "
                "(tool was not invoked); "
                f"got reason={self.decision.reason!r}, allow=True"
            )


Outcome = Allow | Block | Ask | Replace
