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


def test_bottom_toolbar_is_uniformly_monochrome(tmp_path: Path) -> None:
    # Deliberately no red / yellow / green traffic-light anywhere on the
    # bar — operators told us the multi-coloured gradient against the
    # rest of the line being uniform felt noisy. Mono ``ansigray`` only.
    from aura.cli.status_bar import render_bottom_toolbar_html

    for pct_driver in (100, 50_000, 150_000):  # low / mid / high pressure
        s = str(
            render_bottom_toolbar_html(
                model="m",
                input_tokens=pct_driver,
                cache_read_tokens=0,
                context_window=200_000,
                mode="default",
                cwd=tmp_path,
            )
        )
        assert "ansired" not in s
        assert "ansiyellow" not in s
        assert "ansigreen" not in s


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


def test_bottom_toolbar_shows_pinned_estimate_when_cached_is_zero(
    tmp_path: Path,
) -> None:
    # Provider doesn't support prompt caching (cache_read_tokens=0) ⇒
    # fall back to the estimate so the operator still sees a number for
    # the pinned prompt channel. Use ``~`` prefix so it's clear this
    # is an estimate, not a real measurement.
    proj = tmp_path / "proj"
    proj.mkdir()
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="deepseek:glm-5",
            input_tokens=0,
            cache_read_tokens=0,
            context_window=512_000,
            mode="default",
            cwd=proj,
            pinned_estimate_tokens=4_300,
        )
    )
    assert "~4.3k pinned" in s
    # Must NOT also render "cached" — the two are alternatives, not both.
    assert "cached" not in s


def test_bottom_toolbar_prefers_real_cached_over_pinned_estimate(
    tmp_path: Path,
) -> None:
    # When the provider gives us a real number, use it. The estimate
    # is just a fallback — real wins.
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="anthropic:claude-opus-4",
            input_tokens=1000,
            cache_read_tokens=4_100,  # real, from provider
            context_window=200_000,
            mode="default",
            cwd=tmp_path,
            pinned_estimate_tokens=4_300,  # estimate, should be hidden
        )
    )
    assert "4.1k cached" in s
    assert "pinned" not in s


# --------------------------------------------------------------------------
# compact-soon warning (turn input ≥ 80% of window)
# --------------------------------------------------------------------------


def test_warning_shows_above_80_pct(tmp_path: Path) -> None:
    # 85k / 100k = 85% — past the 80% heads-up threshold.
    text = render_status_bar(
        model="m",
        input_tokens=85_000,
        cache_read_tokens=0,
        context_window=100_000,
        mode="default",
        cwd=tmp_path,
    )
    assert "⚠ compact soon" in text.plain


def test_warning_absent_at_79_pct(tmp_path: Path) -> None:
    # 79k / 100k = 79% — just under the threshold, no warning yet.
    text = render_status_bar(
        model="m",
        input_tokens=79_000,
        cache_read_tokens=0,
        context_window=100_000,
        mode="default",
        cwd=tmp_path,
    )
    assert "compact soon" not in text.plain


def test_warning_absent_when_no_input(tmp_path: Path) -> None:
    # 0 input = no useful ratio; elide the warning regardless of window.
    text = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=100_000,
        mode="default",
        cwd=tmp_path,
    )
    assert "compact soon" not in text.plain


def test_warning_shows_above_80_pct_html(tmp_path: Path) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=85_000,
            cache_read_tokens=0,
            context_window=100_000,
            mode="default",
            cwd=tmp_path,
        )
    )
    assert "⚠ compact soon" in s
    assert "ansiyellow" in s


def test_warning_absent_at_79_pct_html(tmp_path: Path) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=79_000,
            cache_read_tokens=0,
            context_window=100_000,
            mode="default",
            cwd=tmp_path,
        )
    )
    assert "compact soon" not in s


def test_warning_absent_when_no_input_html(tmp_path: Path) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=0,
            cache_read_tokens=0,
            context_window=100_000,
            mode="default",
            cwd=tmp_path,
        )
    )
    assert "compact soon" not in s


# --------------------------------------------------------------------------
# wall-clock turn duration (``last_turn_seconds``)
# --------------------------------------------------------------------------


def test_render_status_bar_shows_seconds_with_decimal_under_60s(
    tmp_path: Path,
) -> None:
    text = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        last_turn_seconds=3.4,
    )
    s = text.plain
    assert "3.4s" in s
    # Duration is the final piece so the eye lands on it last.
    assert s.rstrip().endswith("3.4s")


def test_render_status_bar_shows_integer_seconds_at_or_above_60s(
    tmp_path: Path,
) -> None:
    text = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        last_turn_seconds=75.6,
    )
    s = text.plain
    assert "75s" in s
    # No decimal once we're past the minute threshold.
    assert "75.6s" not in s
    # The 60s boundary is inclusive — exactly 60 should also be integer.
    text60 = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=tmp_path,
        last_turn_seconds=60.0,
    )
    assert "60s" in text60.plain
    assert "60.0s" not in text60.plain


def test_render_status_bar_elides_duration_when_zero(tmp_path: Path) -> None:
    # Deterministic basename so tmp_path naming can't bleed into the
    # assertion (pytest's default paths contain the function name).
    proj = tmp_path / "proj"
    proj.mkdir()
    # Default kwarg (0.0) suppresses the piece entirely so the first paint
    # before any turn doesn't read ``... · cwd · 0.0s``.
    text = render_status_bar(
        model="m",
        input_tokens=0,
        cache_read_tokens=0,
        context_window=128_000,
        mode="default",
        cwd=proj,
    )
    s = text.plain
    # cwd (``proj``) is the last piece; no trailing duration should follow.
    assert s.rstrip().endswith("proj")
    assert "0.0s" not in s
    # Be precise: look for a seconds piece, not the literal "0s" which
    # could false-match inside a future cwd name. The full rendered form
    # is ``N.Ns`` or ``Ns``; if none exist, no digits-followed-by-s.
    import re
    assert re.search(r"\d+(?:\.\d+)?s", s) is None


def test_render_bottom_toolbar_html_shows_seconds_with_decimal_under_60s(
    tmp_path: Path,
) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=0,
            cache_read_tokens=0,
            context_window=128_000,
            mode="default",
            cwd=tmp_path,
            last_turn_seconds=3.4,
        )
    )
    assert "3.4s" in s


def test_render_bottom_toolbar_html_shows_integer_seconds_at_or_above_60s(
    tmp_path: Path,
) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html

    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=0,
            cache_read_tokens=0,
            context_window=128_000,
            mode="default",
            cwd=tmp_path,
            last_turn_seconds=75.6,
        )
    )
    assert "75s" in s
    assert "75.6s" not in s


def test_render_bottom_toolbar_html_elides_duration_when_zero(
    tmp_path: Path,
) -> None:
    from aura.cli.status_bar import render_bottom_toolbar_html

    proj = tmp_path / "proj"
    proj.mkdir()
    s = str(
        render_bottom_toolbar_html(
            model="m",
            input_tokens=0,
            cache_read_tokens=0,
            context_window=128_000,
            mode="default",
            cwd=proj,
        )
    )
    assert "0.0s" not in s
    # No ``<digits>s`` duration piece when the kwarg defaults to 0.
    import re
    assert re.search(r">\d+(?:\.\d+)?s<", s) is None
