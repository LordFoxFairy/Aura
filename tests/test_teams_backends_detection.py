"""Detection helpers for the pane backend.

The three predicates have to be cheap, side-effect-free, and easy to
monkeypatch in isolation so the registry's gating logic is testable
without touching the real environment. We exercise both the env-var
branch (``$TMUX``) and the binary-on-PATH branch (``shutil.which``).
"""

from __future__ import annotations

import pytest

from aura.core.teams.backends import detection


def test_detection_returns_false_outside_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``$TMUX`` -> ``is_inside_tmux`` is False (the documented signal)."""
    monkeypatch.delenv("TMUX", raising=False)
    assert detection.is_inside_tmux() is False


def test_detection_returns_true_inside_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    """``$TMUX`` set -> ``is_inside_tmux`` is True regardless of value."""
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,12345,0")
    assert detection.is_inside_tmux() is True


def test_tmux_available_uses_shutil_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tmux_available`` is just ``shutil.which`` truthiness."""
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _name: "/usr/bin/tmux")
    assert detection.tmux_available() is True
    monkeypatch.setattr(_shutil, "which", lambda _name: None)
    assert detection.tmux_available() is False


def test_pane_backend_available_requires_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined gate fails when either predicate is False."""
    monkeypatch.setattr(detection, "is_inside_tmux", lambda: True)
    monkeypatch.setattr(detection, "tmux_available", lambda: True)
    assert detection.pane_backend_available() is True

    monkeypatch.setattr(detection, "is_inside_tmux", lambda: False)
    assert detection.pane_backend_available() is False

    monkeypatch.setattr(detection, "is_inside_tmux", lambda: True)
    monkeypatch.setattr(detection, "tmux_available", lambda: False)
    assert detection.pane_backend_available() is False
