"""Permission mode — how the gate behaves globally.

Spec §3.4. ``default`` is the standard flow (rule → ask → default-deny);
``bypass`` allows everything with a red banner + per-call journal flag,
for scripted / headless runs.

No plan / acceptEdits / dontAsk / auto modes in MVP (§9 non-goals).
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["default", "bypass"]
DEFAULT_MODE: Mode = "default"
