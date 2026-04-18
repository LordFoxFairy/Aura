"""Tests for aura.cli.__main__ entry point."""

from __future__ import annotations

import subprocess
import sys


def test_version_flag_fast_path() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aura" in result.stdout.lower()


def test_help_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aura.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aura" in result.stdout.lower()
