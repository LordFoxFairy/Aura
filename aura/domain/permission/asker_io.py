"""Asker IO contract: the prompt shown to a human and their response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_ASKER_CHOICES: frozenset[str] = frozenset(
    {"yes", "yes-always", "no", "no-always"}
)


@dataclass(frozen=True)
class AskerPrompt:
    tool: str
    args_preview: str
    rule_hint: str
    is_destructive: bool
    request_id: str

    def __post_init__(self) -> None:
        if not self.tool:
            raise ValueError("AskerPrompt requires a non-empty tool name")
        if not self.request_id:
            raise ValueError(
                "AskerPrompt requires a non-empty request_id "
                "(used for IPC correlation)"
            )


@dataclass(frozen=True)
class AskerResponse:
    choice: Literal["yes", "yes-always", "no", "no-always"]
    request_id: str

    def __post_init__(self) -> None:
        if self.choice not in _ASKER_CHOICES:
            raise ValueError(
                f"AskerResponse.choice must be one of {sorted(_ASKER_CHOICES)!r}; "
                f"got {self.choice!r}"
            )
        if not self.request_id:
            raise ValueError(
                "AskerResponse requires a non-empty request_id "
                "(must echo AskerPrompt.request_id for IPC correlation)"
            )
