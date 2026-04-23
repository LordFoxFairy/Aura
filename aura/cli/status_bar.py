"""Per-turn footer rendering — `model · 5.2k tokens · plan mode · cwd`.

Not a true fixed-at-bottom status bar (that would require the unified
prompt-toolkit Application refactor slated for 0.7.0+). Instead this is a
dim one-line footer printed after each turn so the user has continuous
visibility into model / budget / mode without eating alternate-screen real
estate.
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text


def _humanize_tokens(n: int) -> str:
    """1234 -> ``1.2k`` / 1_234_567 -> ``1.2M``. Matches the convention the
    CLI audit line uses so operator scanning across surfaces is consistent."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n/1000:.1f}k"
    return f"{n/1_000_000:.1f}M"


def render_status_bar(
    *,
    model: str | None,
    tokens_used: int,
    mode: str,
    cwd: Path,
) -> Text:
    """Build a single-line dim status line as a ``rich.text.Text``.

    Empty ``model`` elides the model piece; ``mode == "default"`` elides the
    mode indicator (it's the common case, noise otherwise). Always shows
    ``tokens_used`` and the basename of ``cwd``.
    """
    parts: list[str] = []
    if model:
        parts.append(model)
    parts.append(f"{_humanize_tokens(tokens_used)} tokens")
    if mode != "default":
        parts.append(f"{mode} mode")
    parts.append(cwd.name)
    return Text(" · ".join(parts), style="dim")
