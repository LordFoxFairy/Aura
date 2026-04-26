"""Aura desktop bridge — Python side.

Entry points for the Tauri desktop app. Distinct from :mod:`aura.cli`
(the interactive prompt_toolkit REPL) — both are command-line spawnable
Python modules but they're parallel implementations with NO shared
runtime code.

Module map:

- :mod:`aura.desktop.headless` — stdio NDJSON entry the Rust bridge
  spawns. Reads ``{"kind": "prompt", ...}`` / ``{"kind": "permission_response",
  ...}`` requests from stdin, emits one JSON event per line on stdout.
"""
