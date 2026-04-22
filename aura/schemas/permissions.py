"""On-disk permissions schema — neutral leaf type.

Lives in ``aura.schemas`` (the dependency-free leaf layer) so both
``aura.config.schema`` and ``aura.core.permissions.store`` can depend on
it without creating a cycle. See ``aura/schemas/__init__.py`` for the
invariant: nothing under this package imports any other ``aura`` module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["default", "bypass", "plan", "accept_edits"] = "default"
    allow: list[str] = Field(default_factory=list)
    safety_exempt: list[str] = Field(default_factory=list)
