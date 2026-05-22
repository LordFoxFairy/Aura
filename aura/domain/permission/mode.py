"""Permission mode — how the gate behaves globally."""

from __future__ import annotations

from typing import Literal

Mode = Literal["default", "bypass", "plan", "accept_edits"]
DEFAULT_MODE: Mode = "default"
