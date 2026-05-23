"""Permission asker contract — Protocol + reply value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from aura.domain.permission.rule import Rule


@dataclass(frozen=True)
class AskerResponse:
    """Asker reply.

    ``choice == "always"`` iff ``rule is not None``. ``feedback`` is the
    free-text note from Tab-to-amend; appended to ``user_deny`` errors so
    the LLM sees why the user said no.
    """

    choice: Literal["accept", "always", "deny"]
    scope: Literal["project", "session"] = "session"
    rule: Rule | None = None
    feedback: str = ""

    def __post_init__(self) -> None:
        if self.choice == "always" and self.rule is None:
            raise ValueError("choice='always' requires a rule to install")
        if self.choice != "always" and self.rule is not None:
            raise ValueError(
                f"choice={self.choice!r} must not carry a rule"
            )


@runtime_checkable
class PermissionAsker(Protocol):
    """Async oracle returning a user choice for one tool call."""

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse: ...


__all__ = ["AskerResponse", "PermissionAsker"]
