"""On-disk permissions schema + the Phase-1 :class:`Outcome` tagged union.

Lives in ``aura.schemas`` (the dependency-free leaf layer) so both
``aura.config.schema`` and ``aura.core.permissions.store`` can depend on
it without creating a cycle. See ``aura/schemas/__init__.py`` for the
invariant: nothing under this package imports any other ``aura`` module
**at runtime**. The Phase 1 :class:`Outcome` variants reference
``Decision`` and ``ToolResult`` only via :data:`typing.TYPE_CHECKING`
imports + ``from __future__ import annotations`` so the runtime import
graph stays acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aura.schemas.tool import ToolResult

if TYPE_CHECKING:
    # ``Decision`` lives under ``aura.core.permissions.decision``;
    # importing it at runtime would break the ``aura.schemas`` leaf
    # invariant. The annotations below resolve via PEP 563 (string
    # form) — ``__post_init__`` only does attribute access
    # (``decision.allow``), never an ``isinstance`` check, so the
    # actual class is never needed at module import time.
    from aura.core.permissions.decision import Decision


class StatusLineConfig(BaseModel):
    """User-supplied command that overrides the bottom-toolbar render.

    Mirrors claude-code's ``statusLine`` hook: if ``command`` is set and
    ``enabled`` is true, Aura shells out on each toolbar paint, pipes a
    JSON envelope on stdin, and uses the command's stdout as the
    toolbar text (ANSI color codes preserved). Any failure
    (non-zero exit, timeout, crash, empty output) silently falls back
    to the default Aura render — never blocks the REPL.

    Fields are all optional so existing ``settings.json`` files without a
    ``statusline`` section keep working; the whole section itself is
    optional on ``PermissionsConfig``.
    """

    model_config = ConfigDict(extra="forbid")

    command: str = ""
    timeout_ms: int = 500
    enabled: bool = True

    @field_validator("timeout_ms")
    @classmethod
    def _clamp_timeout(cls, v: int) -> int:
        # [50, 5000]ms — below 50ms even a trivial shell exec races the
        # kill path; above 5s the operator would notice the lag and
        # mis-attribute it to the model. Silently clamp rather than
        # raise so a typo doesn't break the REPL.
        if v < 50:
            return 50
        if v > 5000:
            return 5000
        return v

    @property
    def is_active(self) -> bool:
        """True iff a non-empty command is set AND the user hasn't disabled it."""
        return self.enabled and bool(self.command.strip())


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["default", "bypass", "plan", "accept_edits"] = "default"
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    safety_exempt: list[str] = Field(default_factory=list)
    statusline: StatusLineConfig | None = None
    prompt_timeout_sec: float | None = Field(
        default=300.0,
        description=(
            "Seconds to wait for user response on permission prompts "
            "and user_question widgets before treating the non-response "
            "as a denial. None = wait forever (legacy behavior). Default 300 "
            "(5 minutes)."
        ),
    )
    disable_bypass: bool = Field(
        default=False,
        description=(
            "Org-level kill switch for --bypass-permissions / bypass mode. "
            "When true, any attempt to enter bypass mode is refused with a "
            "loud error at startup. Use in shared / CI / compliance environments."
        ),
    )


# ---------------------------------------------------------------------------
# Phase 1 — :class:`Outcome` tagged union (spec §3.2).
#
# The unified pre-tool hook result contract (replaces legacy three-channel triple).
# with one of four variants. The loop pattern-matches on the variant;
# audit consumers read one channel. Hook authors migrate in Tasks 8-10.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Allow:
    """Permission hook chose to let the tool run.

    Carries the :class:`Decision` so audit consumers can see *why* (rule
    match, mode bypass, user accept, etc.). Invariant —
    ``decision.allow`` MUST be ``True``; constructing :class:`Allow`
    with a deny decision is a category error and raises immediately.
    """

    decision: Decision

    def __post_init__(self) -> None:
        if not self.decision.allow:
            raise ValueError(
                "Allow requires a Decision with allow=True; "
                f"got reason={self.decision.reason!r}, allow=False"
            )


@dataclass(frozen=True)
class Block:
    """Permission hook chose to deny the tool call outright.

    The loop appends a synthetic ``ToolMessage`` carrying
    ``decision.audit_line()`` so the model sees the deny reason on the
    next turn. Invariant — ``decision.allow`` MUST be ``False``.
    """

    decision: Decision

    def __post_init__(self) -> None:
        if self.decision.allow:
            raise ValueError(
                "Block requires a Decision with allow=False; "
                f"got reason={self.decision.reason!r}, allow=True"
            )


@dataclass(frozen=True)
class Ask:
    """Permission hook escalated the decision to the user via the asker.

    ``reason`` is a human-readable label rendered above the prompt
    (e.g. ``"destructive bash command"``). Empty / whitespace reasons
    are rejected at construction so the asker widget never has to
    handle an unrenderable label.

    The loop translates the user's :class:`AskerResponse` into a fresh
    :class:`Decision` and recurses into :class:`Allow` or :class:`Block`.
    """

    reason: str

    def __post_init__(self) -> None:
        if not self.reason or not self.reason.strip():
            raise ValueError("Ask requires a non-empty reason")


@dataclass(frozen=True)
class Replace:
    """Permission hook substituted a synthetic result; the tool is NOT invoked.

    Used today by the bash-safety + restrict-tools paths to inject a
    canned error payload back to the model without ever touching the
    underlying tool. Invariants — ``result is not None`` AND
    ``decision.allow`` MUST be ``False`` (the tool was prevented, even
    though we're substituting output rather than raising).
    """

    result: ToolResult
    decision: Decision

    def __post_init__(self) -> None:
        if self.result is None:
            raise ValueError("Replace requires a non-None ToolResult")
        if self.decision.allow:
            raise ValueError(
                "Replace requires a Decision with allow=False "
                "(tool was not invoked); "
                f"got reason={self.decision.reason!r}, allow=True"
            )


# Tagged union — the four variants above are the only legal returns of
# a Phase-1 ``pre_tool`` hook. Pattern-match on the variant in the loop:
#
#     match outcome:
#         case Allow(decision=d):  ...
#         case Block(decision=d):  ...
#         case Ask(reason=r):      ...
#         case Replace(result=r, decision=d): ...
#
Outcome = Allow | Block | Ask | Replace


# ---------------------------------------------------------------------------
# Phase 5 — :class:`AskerPrompt` + :class:`AskerResponse` (spec §6).
#
# The four-state user choice contract (yes / yes-always / no / no-always)
# crossing the asker boundary. Today the same shape is encoded ad-hoc in
# CLI picker, IPC asker (desktop), and subagent auto-deny — Phase 5 Task 3
# formalizes the dataclasses; Tasks 4-6 migrate each asker. The wire
# format on the desktop transport (``aura.permission.request`` /
# ``aura.permission.response``) maps 1:1 to these shapes.
# ---------------------------------------------------------------------------


_ASKER_CHOICES: frozenset[str] = frozenset(
    {"yes", "yes-always", "no", "no-always"}
)


@dataclass(frozen=True)
class AskerPrompt:
    """Everything an asker needs to render the permission prompt.

    Carries display strings (``tool``, ``args_preview``, ``rule_hint``)
    plus the ``is_destructive`` hint the widget uses to flip styling.
    ``request_id`` is the IPC correlation key — the desktop frontend
    matches each ``aura.permission.response`` envelope back to its
    pending request by string equality.

    Invariants enforced at construction so a malformed prompt fails at
    the asker boundary, not deep inside the widget render path:

    - ``tool`` non-empty (the widget uses it as the prompt label).
    - ``request_id`` non-empty (no correlation key = unroutable response).
    """

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
    """The user's four-state choice plus the echoed correlation id.

    The gate (Phase 5 Task 8) translates each ``choice`` value into a
    :class:`Decision` via the strict factories:

    - ``yes``         → ``decision_allow_user_once()``
    - ``yes-always``  → save rule + ``decision_allow_user_always(rule)``
    - ``no``          → ``decision_block_user_once()``
    - ``no-always``   → save deny rule + ``decision_block_rule(rule)``

    Invariants enforced at construction:

    - ``choice`` is one of the four literal values. ``Literal`` only
      constrains static typing; a malformed IPC payload deserialized
      to a foreign string would slip past mypy, so we validate at
      runtime too.
    - ``request_id`` non-empty (echoes the prompt's id verbatim).
    """

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
