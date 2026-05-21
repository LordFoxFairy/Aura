"""Aura desktop — Tauri frontend + Python host process.

This package sits next to ``aura/`` and ``cli/`` at the repo top level.

Sub-packages:

- ``desktop.host`` — Python NDJSON entrypoint spawned by the Tauri Rust
  bridge (``python -m desktop.host.headless``). See ``desktop/host/`` for
  the session service and headless wrapper.

The ``frontend/`` (TS/React) and ``src-tauri/`` (Rust) siblings are not
Python packages; they live in the same directory only because they are
also part of the desktop product surface.
"""
