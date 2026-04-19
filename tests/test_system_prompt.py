"""Tests for aura.core.system_prompt.build_system_prompt."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from aura.core.memory.system_prompt import build_system_prompt


def test_identity_section_mentions_aura(tmp_path: Path) -> None:
    result = build_system_prompt(cwd=tmp_path)
    assert "Aura" in result


def test_environment_section_includes_cwd_and_date(tmp_path: Path) -> None:
    fixed_now = dt.datetime(2026, 4, 17, tzinfo=dt.UTC)
    result = build_system_prompt(cwd=tmp_path, now=fixed_now)
    assert "2026-04-17" in result
    assert str(tmp_path) in result


def test_no_tools_section(tmp_path: Path) -> None:
    result = build_system_prompt(cwd=tmp_path)
    assert "<tools>" not in result


def test_no_aura_md_content_even_when_present(tmp_path: Path) -> None:
    (tmp_path / "AURA.md").write_text("secret-marker-xyz")
    result = build_system_prompt(cwd=tmp_path)
    assert "secret-marker-xyz" not in result
    assert "<project_memory" not in result


def test_registry_kwarg_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_system_prompt(registry="anything", cwd=tmp_path)  # type: ignore[call-arg]
