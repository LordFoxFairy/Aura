"""On-disk permissions schema — neutral leaf type.

Lives in ``aura.schemas`` (the dependency-free leaf layer) so both
``aura.config.schema`` and ``aura.core.permissions.store`` can depend on
it without creating a cycle. See ``aura/schemas/__init__.py`` for the
invariant: nothing under this package imports any other ``aura`` module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
