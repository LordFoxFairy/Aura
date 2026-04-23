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
        input_tokens=5200,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=proj,
    )
    s = text.plain
    # "<anything> mode" indicator absent — only the literal ' mode' suffix
    # fails; cwd basename is "proj", clean.
    assert " mode" not in s
    assert "5.2k/128k" in s
    assert "openai:gpt-4o-mini" in s


def test_plan_mode_shows_plan_indicator(tmp_path: Path) -> None:
    text = render_status_bar(
        model="openai:gpt-4o-mini",
        input_tokens=10,
        cache_read_tokens=0,
        context_window=128_000,
        mode="plan",
        cwd=tmp_path,
    )
    assert "plan mode" in text.plain


def test_accept_edits_mode_shows_indicator(tmp_path: Path) -> None:
    text = render_status_bar(
        model="openai:gpt-4o-mini",
        input_tokens=10,
        cache_read_tokens=0,
        context_window=128_000,
        mode="accept_edits",
        cwd=tmp_path,
    )
    assert "accept_edits mode" in text.plain


def test_empty_model_elides_model_piece(tmp_path: Path) -> None:
    text = render_status_bar(
        model=None,
        input_tokens=42,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    # no model piece leading the line
    assert not text.plain.startswith(" · ")
    assert "42/128k" in text.plain


def test_cwd_name_is_last_piece(tmp_path: Path) -> None:
    text = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    assert text.plain.rstrip().endswith(tmp_path.name)


def test_status_bar_is_dim_styled(tmp_path: Path) -> None:
    text = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    assert text.style == "dim"


# --------------------------------------------------------------------------
# context percentage + cached breakdown
# --------------------------------------------------------------------------
def test_status_bar_shows_context_percentage(tmp_path: Path) -> None:
    # 5400 / 200000 = 2.7% → rounds to integer 2 or 3%. Just assert the
    # percentage token rendered with a '%' sign is present.
    text = render_status_bar(
        model="claude-3-5-sonnet",
        input_tokens=5400,
        cache_read_tokens=0,
        context_window=200_000,
        mode="default",
        cwd=tmp_path,
    )
    s = text.plain
    assert "5.4k/200k" in s
    # Parenthesized percentage piece for nonzero input.
    assert "%" in s
    assert "(" in s and ")" in s


def test_status_bar_shows_cached_when_nonzero(tmp_path: Path) -> None:
    text = render_status_bar(
        model="claude-3-5-sonnet",
        input_tokens=5400,
        cache_read_tokens=34_000,
        context_window=200_000,
        mode="default",
        cwd=tmp_path,
    )
    s = text.plain
    assert "+34.0k cached" in s or "+34k cached" in s


def test_status_bar_omits_cached_when_zero(tmp_path: Path) -> None:
    # Use a deterministic subdir basename so pytest's tmp_path (which
    # sometimes contains the test-function name, including the substring
    # "cached") can't pollute the assertion.
    proj = tmp_path / "proj"
    proj.mkdir()
    text = render_status_bar(
        model="claude-3-5-sonnet",
        input_tokens=5400,
        cache_read_tokens=0,
        context_window=200_000,
        mode="default",
        cwd=proj,
    )
    assert "cached" not in text.plain


def test_status_bar_zero_input_no_percent(tmp_path: Path) -> None:
    # Fresh start: show 0/window with no confusing "0%" pair.
    text = render_status_bar(
        model="openai:gpt-4o-mini",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
    )
    s = text.plain
    assert "0/128k" in s
    assert "%" not in s


def test_status_bar_humanizes_large_window(tmp_path: Path) -> None:
    # 200_000 must render as "200k", not "200000".
    text = render_status_bar(
        model="claude",
        input_tokens=1000,
        cache_read_tokens=0,
        context_window=200_000,
        mode="default",
        cwd=tmp_path,
    )
    assert "200k" in text.plain
    assert "200000" not in text.plain


# --------------------------------------------------------------------------
# Visual bottom-toolbar — progress bar + color coding
# --------------------------------------------------------------------------


def test_context_bar_empty() -> None:
    from aura.cli.status_bar import _render_context_bar
    assert _render_context_bar(0) == "░░░░░░░░░░"


def test_context_bar_half() -> None:
    from aura.cli.status_bar import _render_context_bar
    assert _render_context_bar(50) == "█████░░░░░"


def test_context_bar_full() -> None:
    from aura.cli.status_bar import _render_context_bar
    assert _render_context_bar(100) == "██████████"


def test_context_bar_clamps_over_100() -> None:
    from aura.cli.status_bar import _render_context_bar
    # Malformed input (e.g. computed pct rounding weirdness) must not crash
    # and must not emit more blocks than width.
    out = _render_context_bar(150)
    assert len(out) == 10
    assert out == "██████████"


def test_pct_color_green_below_30() -> None:
    from aura.cli.status_bar import _pct_color_tag
    assert _pct_color_tag(0) == "ansigreen"
    assert _pct_color_tag(29) == "ansigreen"


def test_pct_color_yellow_30_to_59() -> None:
    from aura.cli.status_bar import _pct_color_tag
    assert _pct_color_tag(30) == "ansiyellow"
    assert _pct_color_tag(59) == "ansiyellow"


def test_pct_color_red_60_and_above() -> None:
    from aura.cli.status_bar import _pct_color_tag
    assert _pct_color_tag(60) == "ansired"
    assert _pct_color_tag(95) == "ansired"


def test_bottom_toolbar_html_contains_model_tokens_bar_cached(
    tmp_path: Path,
) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html
    out = render_bottom_toolbar_html(
        model="claude-3-5-sonnet",
        input_tokens=5400,
        cache_read_tokens=34_000,
        context_window=200_000,
        mode="default",
        cwd=tmp_path,
    )
    s = str(out)
    assert "claude-3-5-sonnet" in s
    assert "5.4k/200k" in s
    assert "[" in s and "]" in s     # bar
    assert "3%" in s                  # 5400/200000 ≈ 2.7% → rounds to 3%
    assert "34.0k cached" in s or "34k cached" in s
    assert tmp_path.name in s


def test_bottom_toolbar_html_omits_cached_when_zero(tmp_path: Path) -> None:
    # Use a deterministic subdir basename that can't match "cached".
    proj = tmp_path / "proj"
    proj.mkdir()
    from aura.cli.status_bar import render_bottom_toolbar_html
    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=100,
            cache_read_tokens=0,
            context_window=128_000,
            mode="default",
            cwd=proj,
        )
    )
    assert "cached" not in s


def test_bottom_toolbar_color_tag_at_high_pressure(tmp_path: Path) -> None:
    # 150k/200k = 75% → red.
    from aura.cli.status_bar import render_bottom_toolbar_html
    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=150_000,
            cache_read_tokens=0,
            context_window=200_000,
            mode="default",
            cwd=tmp_path,
        )
    )
    assert "ansired" in s
    assert "ansiyellow" not in s
    assert "ansigreen" not in s
