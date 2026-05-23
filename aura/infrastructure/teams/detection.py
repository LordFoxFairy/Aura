"""Environment detection for the pane backend; uncached so session switches are noticed."""

from __future__ import annotations

import os
import shutil


def is_inside_tmux() -> bool:
    return bool(os.environ.get("TMUX"))


def tmux_available() -> bool:
    return shutil.which("tmux") is not None


def pane_backend_available() -> bool:
    return is_inside_tmux() and tmux_available()


__all__ = ["is_inside_tmux", "pane_backend_available", "tmux_available"]
