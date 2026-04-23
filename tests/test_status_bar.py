"""Tests for aura.cli.status_bar."""

from __future__ import annotations

from pathlib import Path

from aura.cli.status_bar import _humanize_tokens, render_status_bar


# --------------------------------------------------------------------------
# humanize_tokens
# --------------------------------------------------------------------------
def test_humanize_tokens_below_1k_is_literal() -> None:
    assert _humanize_tokens(0) == "0"
    assert _humanize_tokens(999) == "999"


def test_humanize_tokens_below_1m_uses_k_suffix() -> None:
    assert _humanize_tokens(1000) == "1.0k"
    assert _humanize_tokens(1234) == "1.2k"
    assert _humanize_tokens(999_000).endswith("k")


def test_humanize_tokens_million_uses_m_suffix() -> None:
    assert _humanize_tokens(1_000_000) == "1.0M"
    assert _humanize_tokens(2_500_000) == "2.5M"


# --------------------------------------------------------------------------
# render_status_bar
# --------------------------------------------------------------------------
def test_default_mode_omits_mode_indicator(tmp_path: Path) -> None:
    # Use a subdir with a deterministic name to avoid pytest's tmp_path
    # basename leaking "mode" into the assertion.
    proj = tmp_path / "proj"
    proj.mkdir()
    text = render_status_bar(
        model="openai:gpt-4o-mini",
        tokens_used=5200,
        mode="default",
        cwd=proj,
    )
    s = text.plain
    # "<anything> mode" indicator absent — only the literal ' mode' suffix
    # fails; cwd basename is "proj", clean.
    assert " mode" not in s
    assert "5.2k tokens" in s
    assert "openai:gpt-4o-mini" in s


def test_plan_mode_shows_plan_indicator(tmp_path: Path) -> None:
    text = render_status_bar(
        model="openai:gpt-4o-mini",
        tokens_used=10,
        mode="plan",
        cwd=tmp_path,
    )
    assert "plan mode" in text.plain


def test_accept_edits_mode_shows_indicator(tmp_path: Path) -> None:
    text = render_status_bar(
        model="openai:gpt-4o-mini",
        tokens_used=10,
        mode="accept_edits",
        cwd=tmp_path,
    )
    assert "accept_edits mode" in text.plain


def test_empty_model_elides_model_piece(tmp_path: Path) -> None:
    text = render_status_bar(
        model=None,
        tokens_used=42,
        mode="default",
        cwd=tmp_path,
    )
    # no model piece leading the line
    assert not text.plain.startswith(" · ")
    assert "42 tokens" in text.plain


def test_cwd_name_is_last_piece(tmp_path: Path) -> None:
    text = render_status_bar(
        model="m",
        tokens_used=0,
        mode="default",
        cwd=tmp_path,
    )
    assert text.plain.rstrip().endswith(tmp_path.name)


def test_status_bar_is_dim_styled(tmp_path: Path) -> None:
    text = render_status_bar(
        model="m", tokens_used=0, mode="default", cwd=tmp_path,
    )
    assert text.style == "dim"
