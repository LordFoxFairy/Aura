"""Status-bar rendering — two surfaces, same data.

Two callers, two output formats:

1. ``render_status_bar`` → ``rich.text.Text``. Used for one-off/transcript
   printing (tests, non-TTY fallback). Plain-text with rich style.
2. ``render_bottom_toolbar_html`` → ``prompt_toolkit.formatted_text.HTML``.
   Used as the *live* bottom bar under the REPL prompt; pt re-invokes the
   toolbar callable on every render so the numbers auto-update between
   turns. Color-coded percentage bar so operators can SEE context
   pressure without reading the number.

Design notes
------------
- ``input_tokens`` is the *last turn's* prompt size — the dynamic part of
  the context bill. Cumulative totals live on ``state.custom['_token_stats']``
  but aren't shown here: operators care about "how much did THIS turn cost",
  not an ever-growing running total.
- ``cache_read_tokens`` is the portion of this turn's input that hit the
  prompt cache. Shown separately because it's effectively free / pinned —
  calling it out prevents operators from mistaking a 40k cached-prefix
  turn for an expensive one.
- ``context_window`` is the model's declared max context. Resolve it via
  :func:`aura.core.llm.get_context_window`.
- Progress bar color gradient: green < 30% < yellow < 60% < red. Mirrors
  the operator's intuition ("I can still breathe" / "watch it" / "about
  to overflow — compact!") so the bar color IS the message.
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.formatted_text import HTML
from rich.text import Text


def _humanize_tokens(n: int) -> str:
    """1234 -> ``1.2k`` / 1_234_567 -> ``1.2M``. Matches the convention the
    CLI audit line uses so operator scanning across surfaces is consistent."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1000:.1f}k"
    return f"{n/1_000_000:.1f}M"


def _humanize_window(n: int) -> str:
    """Context windows are always round thousands (128_000, 200_000, …)
    so we strip the ``.0`` decimal to keep the line compact.
    ``128_000`` → ``128k``, not ``128.0k``.
    """
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        whole = n // 1000
        # Only collapse to bare 'k' form when it's a clean thousand. Weird
        # values (e.g. 16_385 for gpt-3.5) fall back to the 1-decimal form.
        if n % 1000 == 0:
            return f"{whole}k"
        return f"{n/1000:.1f}k"
    return f"{n/1_000_000:.1f}M"


def render_status_bar(
    *,
    model: str | None,
    input_tokens: int,
    cache_read_tokens: int,
    context_window: int,
    mode: str,
    cwd: Path,
) -> Text:
    """Build a single-line dim status line as a ``rich.text.Text``.

    Format pieces joined by ` · `:
      1. ``model`` — elided when ``None``
      2. ``<input>/<window>`` (+ ``(X%)`` when input > 0)
      3. ``+<cached>k cached`` — elided when ``cache_read_tokens == 0``
      4. ``<mode> mode`` — elided when ``mode == "default"``
      5. basename of ``cwd``
    """
    parts: list[str] = []
    if model:
        parts.append(model)

    live = _humanize_tokens(input_tokens)
    window = _humanize_window(context_window)
    if input_tokens > 0 and context_window > 0:
        pct = round(input_tokens / context_window * 100)
        # Never display 0% for nonzero input — the whole point of the
        # indicator is "you've spent something"; floor to 1%.
        if pct < 1:
            pct = 1
        parts.append(f"{live}/{window} ({pct}%)")
    else:
        parts.append(f"{live}/{window}")

    if cache_read_tokens > 0:
        parts.append(f"+{_humanize_tokens(cache_read_tokens)} cached")

    if mode != "default":
        parts.append(f"{mode} mode")

    parts.append(cwd.name)
    return Text(" · ".join(parts), style="dim")


# ---------------------------------------------------------------------------
# Visual bottom-toolbar rendering (prompt_toolkit HTML)
# ---------------------------------------------------------------------------


def _render_context_bar(pct: int, width: int = 10) -> str:
    """ASCII block-bar: ``████░░░░░░`` at 40%.

    ``pct`` is clamped to [0, 100]. ``width`` fixed at 10 so 1 block = 10%.
    Chosen over percent-only display because a bar reads at glance — the
    operator sees context pressure without reading the number.
    """
    pct_clamped = max(0, min(100, pct))
    filled = round(pct_clamped / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _pct_color_tag(pct: int) -> str:
    """Map a percentage to a prompt_toolkit ANSI color tag.

    Gradient: green (< 30) < yellow (< 60) < red. Thresholds picked so the
    color flips BEFORE the user is in actual trouble — yellow at 30% is a
    'start thinking about compact' signal; red at 60% is 'compact now'.
    """
    if pct < 30:
        return "ansigreen"
    if pct < 60:
        return "ansiyellow"
    return "ansired"


def render_bottom_toolbar_html(
    *,
    model: str | None,
    input_tokens: int,
    cache_read_tokens: int,
    context_window: int,
    mode: str,
    cwd: Path,
) -> HTML:
    """Build the live bottom-toolbar shown under the REPL prompt.

    Format:  ``model · 3.2k/128k [████░░░░░░] 2% · +34k cached · cwd``

    The progress-bar segment is color-coded by ``_pct_color_tag``. Other
    pieces are dim. ``mode`` piece omitted when ``default``.
    """
    live = _humanize_tokens(input_tokens)
    window = _humanize_window(context_window)
    pct = 0
    if input_tokens > 0 and context_window > 0:
        pct = max(1, round(input_tokens / context_window * 100))
    bar = _render_context_bar(pct)
    color = _pct_color_tag(pct)

    pieces: list[str] = []
    if model:
        pieces.append(f"<ansigray>{model}</ansigray>")
    # Bar + number + pct on one segment; only the bar gets the color accent
    # so the rest stays in a uniform dim palette.
    pieces.append(
        f"<ansigray>{live}/{window}</ansigray> "
        f"<{color}>[{bar}]</{color}> "
        f"<ansigray>{pct}%</ansigray>"
    )
    if cache_read_tokens > 0:
        pieces.append(
            f"<ansigray>+{_humanize_tokens(cache_read_tokens)} cached</ansigray>"
        )
    if mode != "default":
        pieces.append(f"<ansigray>{mode} mode</ansigray>")
    pieces.append(f"<ansigray>{cwd.name}</ansigray>")

    return HTML(" · ".join(pieces))
