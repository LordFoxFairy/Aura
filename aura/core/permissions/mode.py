"""Permission mode — how the gate behaves globally.

Spec §3.4, extended 2026-04-21 to the 4-mode claude-code parity set:

- ``default`` — rule → ask → default-deny (the standard flow).
- ``bypass`` — allow everything not blocked by safety; loud banner + per-call
  journal flag; for scripted / headless runs.
- ``plan`` — dry-run every tool call. Safety still fires first; any tool that
  safety permits is then blocked with ``plan_mode_blocked`` so the user sees
  what the agent *would* have done without any side effects.
- ``accept_edits`` — auto-allow the edit-family tools (``read_file``,
  ``write_file``, ``edit_file``) when safety permits; every other tool
  (``bash``, ``web_fetch``, etc.) still flows through the normal rule/ask
  path. Useful for pair-programming sessions where file edits are expected
  and only side-effecting tools need confirmation.

Safety is a separate axis from mode — NO mode bypasses the safety list.
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["default", "bypass", "plan", "accept_edits"]
DEFAULT_MODE: Mode = "default"
