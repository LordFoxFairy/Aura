"""Status-bar rendering — two surfaces, same data.

Two callers, two output formats:

1. ``render_status_bar`` → ``rich.text.Text``. Used for one-off/transcript
   printing (tests, non-TTY fallback). Plain-text with rich style.
2. ``render_bottom_toolbar_html`` → ``prompt_toolkit.formatted_text.HTML``.
   Used as the *live* bottom bar under the REPL prompt; pt re-invokes the
   toolbar callable on every render so the numbers auto-update between
   turns. Uniformly dim (``ansigray``) — no colored traffic-light gradient:
   operators told us the multi-colored bar felt noisy against everything
   else being one tone.

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
- ``pinned_estimate_tokens`` is a local char-count estimate of the pinned
  prompt prefix (system msg + memory + tool schemas). Shown BEFORE the
  first turn, AND when the provider doesn't support prompt caching
  (e.g. deepseek ⇒ ``cache_read_tokens`` is always 0). Prefixed with
  ``~`` so operators can tell it apart from a real ``cached`` number.
- ``context_window`` is the model's declared max context. Resolve it via
  :func:`aura.core.llm.get_context_window`.
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
    pinned_estimate_tokens: int = 0,
) -> Text:
    """Build a single-line dim status line as a ``rich.text.Text``.

    Format pieces joined by ` · `:
      1. ``model`` — elided when ``None``
      2. ``<input>/<window>`` (+ ``(X%)`` when input > 0)
      3. ``+<cached>k cached`` — when ``cache_read_tokens > 0``
         (real measurement from provider); otherwise
         ``~<pinned>k pinned`` when the estimate is non-zero. Same
         pinned-vs-cached resolution as the bottom toolbar.
      4. ``<mode> mode`` — elided when ``mode == "default"``
      5. basename of ``cwd``

    Used both for (a) non-TTY fallback in place of the pt toolbar and
    (b) post-turn checkpoint printed after every response so the
    operator still sees the last-turn stats while the pt toolbar is
    hidden during streaming.
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
    elif pinned_estimate_tokens > 0:
        parts.append(f"~{_humanize_tokens(pinned_estimate_tokens)} pinned")

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


def render_bottom_toolbar_html(
    *,
    model: str | None,
    input_tokens: int,
    cache_read_tokens: int,
    context_window: int,
    mode: str,
    cwd: Path,
    pinned_estimate_tokens: int = 0,
) -> HTML:
    """Build the live bottom-toolbar shown under the REPL prompt.

    Format:  ``model · 3.2k/128k [████░░░░░░] 2% · ~4.3k pinned · cwd``

    Everything is uniformly dim (``ansigray``) — operators asked for
    monochrome after the earlier traffic-light gradient (green / yellow /
    red bar) felt noisy against the rest of the line. ``mode`` piece
    omitted when ``default``.

    Pinned-vs-cached resolution:
      - If the provider reported ``cache_read_tokens > 0`` this turn,
        show the real number as ``+Xk cached``.
      - Otherwise, if a ``pinned_estimate_tokens`` was passed (computed
        at Agent init from char counts), show ``~Xk pinned`` — prefix
        ``~`` signals "estimate, not a measurement".
      - Operators on providers that don't cache (e.g. deepseek) will
        therefore ALWAYS see a pinned-channel number, not a zero.
    """
    live = _humanize_tokens(input_tokens)
    window = _humanize_window(context_window)
    pct = 0
    if input_tokens > 0 and context_window > 0:
        pct = max(1, round(input_tokens / context_window * 100))
    bar = _render_context_bar(pct)

    pieces: list[str] = []
    if model:
        pieces.append(f"<ansigray>{model}</ansigray>")
    pieces.append(f"<ansigray>{live}/{window} [{bar}] {pct}%</ansigray>")
    if cache_read_tokens > 0:
        pieces.append(
            f"<ansigray>+{_humanize_tokens(cache_read_tokens)} cached</ansigray>"
        )
    elif pinned_estimate_tokens > 0:
        pieces.append(
            f"<ansigray>~{_humanize_tokens(pinned_estimate_tokens)} pinned</ansigray>"
        )
    if mode != "default":
        pieces.append(f"<ansigray>{mode} mode</ansigray>")
    pieces.append(f"<ansigray>{cwd.name}</ansigray>")

    return HTML(" · ".join(pieces))
