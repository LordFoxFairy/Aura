"""Generic scrollable, paginated, filter-able item picker for the REPL.

Built on prompt_toolkit 3.x. Two surfaces consume it today:

* The slash-command picker (``aura.cli.completion``) — modal, runs while
  the user is typing ``/foo`` and replaces the dropdown.
* The resume-session picker (``aura.cli.commands`` / ``--resume``) —
  modal, picks one session out of a possibly-very-long list.

Behavior mirrors claude-code's ``CustomSelect`` (TS) widget:

- Bounded viewport (``page_size``); cursor scrolls the window when it
  reaches an edge.
- Up/Down arrow keys with wrap-around; Page Up / Page Down by viewport;
  Home / End jump to first / last.
- Width-aware ``…`` truncation — the label gets at most "available
  width minus marker minus sublabel column".
- Optional incremental substring filter (``enable_filter=True``) that
  re-narrows the visible set as the user types.
- ``Enter`` selects, ``Esc`` (or ``Ctrl-C``) cancels and yields ``None``.
- ``selected`` row uses the ``picker.selected`` style class so terminal
  themes can reskin it without code changes — never raw ANSI escapes.

The widget is a self-contained ``Application``: ``run_picker`` builds it,
runs it, and returns. Callers don't share UI state with it — they just
get a :class:`PickerResult` back when the user picks (or cancels).

Implementation note: we use ``Application(full_screen=False)`` so the
modal renders in-line above the existing scrollback. Full-screen would
clear the user's REPL surface (welcome panel, prior turns) and re-paint
it on exit; the in-line modal slides up cleanly from below the prompt
and the rest of the terminal stays put.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

#: Default viewport size. Eight rows fits comfortably below most prompts
#: without dominating the terminal — matches claude-code's default
#: ``visibleOptionCount`` for ``CustomSelect``.
DEFAULT_PAGE_SIZE: int = 8

#: Glyph painted at the start of the focused row. ``❯`` (U+276F) renders
#: cleanly on every terminal we ship for; falling back to ``>`` on dumb
#: terminals isn't worth the conditional — pt's renderer transparently
#: degrades unsupported codepoints.
_CURSOR_GLYPH: str = "❯"

#: Width of the cursor column (cursor glyph + one trailing space) used
#: for budget calculations. Pre-computed so the per-row layout work
#: doesn't re-call :func:`get_cwidth` on every render.
_CURSOR_COL_WIDTH: int = get_cwidth(_CURSOR_GLYPH) + 1

#: Default Style mapping for the picker. Callers can pass in their own
#: Style and we'll merge over it (this dict acts as the floor — a
#: caller's overrides win). Kept conservative: ``ansibrightblue`` for
#: the cursor + selected row stays legible on both light and dark
#: terminal themes.
_DEFAULT_STYLE: Style = Style.from_dict({
    "picker.title": "bold",
    "picker.cursor": "ansibrightblue bold",
    "picker.selected": "reverse",
    "picker.label": "",
    "picker.sublabel": "ansibrightblack",
    "picker.footer": "ansibrightblack",
    "picker.footer.match": "ansiyellow",
    "picker.empty": "ansibrightblack italic",
    "picker.filter.prompt": "ansibrightblack",
})


class PickerItem(Protocol):
    """Structural type for items the picker renders.

    Every item exposes:

    - ``label`` — primary text, truncated to fit the row.
    - ``sublabel`` — optional dim secondary text shown right-aligned
      within the row (e.g. "2 hours ago"). ``None`` to omit.
    - ``value`` — opaque payload returned to the caller on selection.

    Declared as ``@property`` rather than plain class attributes so
    frozen dataclasses (which expose their fields as read-only at the
    type level) satisfy the protocol. mypy treats a settable attribute
    on a Protocol as "the implementor must accept assignment" — frozen
    dataclasses don't, hence the property form.
    """

    @property
    def label(self) -> str: ...

    @property
    def sublabel(self) -> str | None: ...

    @property
    def value(self) -> object: ...


@dataclass(frozen=True)
class SimplePickerItem:
    """Concrete :class:`PickerItem` for callers that don't need their own.

    Frozen so callers can hash / set-add / re-use items safely. Fields
    match the Protocol above 1:1.
    """

    label: str
    sublabel: str | None
    value: object


@dataclass(frozen=True)
class PickerResult:
    """Outcome of one :func:`run_picker` call.

    ``item`` is ``None`` iff the user cancelled (Esc / Ctrl+C / empty
    list short-circuit). Callers can pattern-match on that:

    >>> result = await run_picker(items, title="...")
    >>> if result.item is None: return
    >>> picked = result.item.value
    """

    item: PickerItem | None


def _terminal_width(default: int = 80) -> int:
    """Return the current terminal column count, with a sensible floor.

    ``shutil.get_terminal_size`` honors COLUMNS env var first, then
    queries the tty. We clamp to ``[40, 200]``: anything below 40 cols
    is too narrow for two-column rendering anyway and we'd rather show
    a tightly-truncated row than wrap; anything above 200 is mostly
    blank space waste — claude-code caps wide terminals the same way.
    """
    try:
        return max(40, min(200, shutil.get_terminal_size((default, 24)).columns))
    except OSError:
        return default


def _truncate_to_width(text: str, max_cols: int) -> str:
    """Truncate ``text`` so its rendered width fits in ``max_cols``.

    Walks the string column-by-column using :func:`get_cwidth` so CJK /
    emoji (which take two columns) are accounted for. When the source
    overflows we leave one column for the ``…`` glyph; no ellipsis is
    added when the text already fits.
    """
    if max_cols <= 0:
        return ""
    if get_cwidth(text) <= max_cols:
        return text
    # Reserve one column for the ellipsis glyph.
    budget = max_cols - 1
    if budget <= 0:
        return "…"
    out: list[str] = []
    used = 0
    for ch in text:
        w = get_cwidth(ch)
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def _matches_filter(item: PickerItem, needle: str) -> bool:
    """Case-insensitive substring match against label + sublabel.

    Empty needle matches every item. The sublabel is searched too so a
    user can filter the resume picker by typing a relative-time
    fragment ("hour", "minute") or session-id substring.
    """
    if not needle:
        return True
    lo = needle.lower()
    if lo in item.label.lower():
        return True
    sub = item.sublabel or ""
    return lo in sub.lower()


class _PickerState:
    """Mutable state owned by one running picker Application.

    Kept off the module level so concurrent pickers (a hypothetical
    nested invocation; not a current path) wouldn't trample each other.
    Holds:

    - ``items`` — the full input list, never mutated.
    - ``filter_text`` — current filter string; updated by the input
      buffer's ``on_text_changed`` callback.
    - ``cursor`` — index into the **filtered** list of the focused row.
    - ``viewport_top`` — index into the filtered list of the topmost
      row currently rendered.
    """

    def __init__(
        self,
        items: Sequence[PickerItem],
        page_size: int,
        initial_filter: str,
    ) -> None:
        self.items: tuple[PickerItem, ...] = tuple(items)
        self.page_size = page_size
        self.filter_text = initial_filter
        self.cursor = 0
        self.viewport_top = 0

    def visible(self) -> list[PickerItem]:
        """Return the items currently passing the filter."""
        return [it for it in self.items if _matches_filter(it, self.filter_text)]

    def clamp_after_filter_change(self) -> None:
        """Re-anchor the cursor / viewport after the filter narrows the list.

        Called whenever the filter string changes. Keeps the cursor
        within ``[0, len(visible)-1]`` and slides the viewport so the
        cursor is always inside it.
        """
        n = len(self.visible())
        if n == 0:
            self.cursor = 0
            self.viewport_top = 0
            return
        if self.cursor >= n:
            self.cursor = n - 1
        # Slide viewport so cursor is in-frame.
        if self.cursor < self.viewport_top:
            self.viewport_top = self.cursor
        if self.cursor >= self.viewport_top + self.page_size:
            self.viewport_top = self.cursor - self.page_size + 1
        self.viewport_top = max(0, min(self.viewport_top, max(0, n - self.page_size)))

    # --- Cursor movement helpers. Return None — they mutate self in-place
    # and the caller invalidates the Application.

    def move_down(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        # Wrap around: pressing Down on the last row jumps to the first.
        self.cursor = (self.cursor + 1) % n
        if self.cursor == 0:
            # Wrapped — reset viewport to top.
            self.viewport_top = 0
            return
        if self.cursor >= self.viewport_top + self.page_size:
            self.viewport_top = self.cursor - self.page_size + 1

    def move_up(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        if self.cursor == 0:
            # Wrap to last; jump viewport to bottom.
            self.cursor = n - 1
            self.viewport_top = max(0, n - self.page_size)
            return
        self.cursor -= 1
        if self.cursor < self.viewport_top:
            self.viewport_top = self.cursor

    def page_down(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        self.cursor = min(n - 1, self.cursor + self.page_size)
        self.viewport_top = max(
            0, min(self.cursor - self.page_size + 1, n - self.page_size),
        )
        if self.viewport_top < 0:
            self.viewport_top = 0

    def page_up(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        self.cursor = max(0, self.cursor - self.page_size)
        self.viewport_top = min(self.viewport_top, self.cursor)

    def home(self) -> None:
        self.cursor = 0
        self.viewport_top = 0

    def end(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        self.cursor = n - 1
        self.viewport_top = max(0, n - self.page_size)


def _render_items(state: _PickerState) -> FormattedText:
    """Render the current viewport into a pt FormattedText fragment list.

    Width-aware: each row's label gets the available terminal width minus
    the cursor column minus the sublabel column (when present) minus a
    one-space separator. Sublabels render right-aligned via spaces so
    timestamps line up cleanly across rows.
    """
    visible = state.visible()
    if not visible:
        return FormattedText([("class:picker.empty", "  (no matches)\n")])

    fragments: list[tuple[str, str]] = []
    width = _terminal_width()
    # Top of viewport — clamp again defensively (state should already
    # have done this; belt-and-braces against a mid-render mutation).
    top = max(0, min(state.viewport_top, max(0, len(visible) - state.page_size)))
    bottom = min(len(visible), top + state.page_size)

    # Indicator row when there are items above the viewport.
    if top > 0:
        fragments.append(
            ("class:picker.footer", f"  ({top} more above)\n"),
        )

    for idx in range(top, bottom):
        item = visible[idx]
        is_cursor = idx == state.cursor

        sublabel = item.sublabel or ""
        sublabel_w = get_cwidth(sublabel) if sublabel else 0
        # Reserve one space between label and sublabel when both are
        # present; nothing reserved when there's no sublabel.
        gap = 1 if sublabel else 0
        label_budget = max(1, width - _CURSOR_COL_WIDTH - sublabel_w - gap - 1)
        label = _truncate_to_width(item.label, label_budget)

        # Build the row. The cursor glyph is its own fragment so styles
        # apply independently of the label color.
        row_style = "class:picker.selected" if is_cursor else ""

        if is_cursor:
            fragments.append(("class:picker.cursor", _CURSOR_GLYPH + " "))
        else:
            fragments.append((row_style, " " * _CURSOR_COL_WIDTH))

        # Pad label to its budget so the sublabel column lines up.
        label_padded = label + " " * max(0, label_budget - get_cwidth(label))
        label_class = (
            "class:picker.selected" if is_cursor else "class:picker.label"
        )
        fragments.append((label_class, label_padded))

        if sublabel:
            fragments.append((row_style, " " * gap))
            sub_class = (
                "class:picker.selected" if is_cursor else "class:picker.sublabel"
            )
            fragments.append((sub_class, sublabel))

        fragments.append(("", "\n"))

    # Indicator row when there are items below the viewport.
    remaining = len(visible) - bottom
    if remaining > 0:
        fragments.append(
            ("class:picker.footer", f"  ({remaining} more below)\n"),
        )

    return FormattedText(fragments)


def _build_application(
    state: _PickerState,
    *,
    title: str,
    enable_filter: bool,
    total_count: int,
    style: Style,
) -> Application[PickerResult]:
    """Assemble the Application that drives one picker invocation.

    The layout is a vertical stack:

      ┌── title bar ──────────┐
      │ <title>               │
      ├── filter input ───────┤  (only when enable_filter=True)
      │ Filter: ___________   │
      ├── items window ───────┤
      │ ❯ item 1            │
      │   item 2  · sublabel  │
      │   ...                 │
      ├── footer hint ────────┤
      │ ↑/↓ navigate · Enter  │
      └───────────────────────┘
    """
    filter_buffer = Buffer(
        multiline=False,
        # Cursor blinks inside the filter input so the user sees where
        # their typing lands. Default behavior — pt blinks the buffer's
        # cursor automatically.
    )
    if state.filter_text:
        filter_buffer.text = state.filter_text

    def _on_filter_changed(_buf: Buffer) -> None:
        state.filter_text = filter_buffer.text
        state.clamp_after_filter_change()

    filter_buffer.on_text_changed += _on_filter_changed

    def _items_text() -> FormattedText:
        return _render_items(state)

    def _title_text() -> FormattedText:
        return FormattedText([("class:picker.title", title + "\n")])

    def _footer_text() -> FormattedText:
        visible_n = len(state.visible())
        parts: list[tuple[str, str]] = []
        # Counter — "showing X of Y" when filtered; "X items" otherwise.
        if state.filter_text:
            parts.append(
                (
                    "class:picker.footer.match",
                    f"  {visible_n} matching of {total_count}",
                ),
            )
        elif total_count > len(state.items):
            # The caller pre-truncated the input list before handing it
            # to us (resume picker shows top-50 of all-sessions). Tell
            # the user there's more below the surface, hint them at the
            # filter as the way to dig deeper.
            parts.append(
                (
                    "class:picker.footer",
                    f"  showing {len(state.items)} of {total_count}"
                    " — type to filter",
                ),
            )
        else:
            parts.append(
                ("class:picker.footer", f"  {visible_n} items"),
            )
        parts.append(
            (
                "class:picker.footer",
                "    ↑/↓ navigate · Enter select · Esc cancel",
            ),
        )
        parts.append(("", "\n"))
        return FormattedText(parts)

    title_window = Window(
        FormattedTextControl(_title_text), height=1, dont_extend_height=True,
    )
    items_window = Window(
        FormattedTextControl(_items_text),
        # Reserve room for above/below indicator lines (+2). Cap so the
        # picker never grows past its viewport — pt would otherwise
        # extend the window when the items text overflowed.
        height=Dimension.exact(state.page_size + 2),
        dont_extend_height=True,
    )
    footer_window = Window(
        FormattedTextControl(_footer_text),
        height=1,
        dont_extend_height=True,
    )

    # Filter row: a single-line input. Implemented as VSplit so the
    # ``filter:`` prefix sits on the same row as the editable buffer.
    from prompt_toolkit.layout.containers import VSplit
    filter_input_window = Window(
        BufferControl(buffer=filter_buffer),
        height=1,
        dont_extend_height=True,
    )
    filter_row = ConditionalContainer(
        VSplit([
            Window(
                FormattedTextControl(
                    FormattedText([
                        ("class:picker.filter.prompt", "filter: "),
                    ]),
                ),
                width=Dimension.exact(get_cwidth("filter: ")),
                dont_extend_width=True,
                height=1,
            ),
            filter_input_window,
        ]),
        # ConditionalContainer wants a Filter — pass a bool literal so
        # pt's ``to_filter`` shortcut converts it to the correct Always
        # / Never sentinel. A plain Python bool is the documented input.
        filter=bool(enable_filter),
    )

    layout = Layout(
        HSplit([
            title_window,
            items_window,
            filter_row,
            footer_window,
        ]),
        focused_element=filter_input_window if enable_filter else items_window,
    )

    kb = KeyBindings()

    def _selected_item() -> PickerItem | None:
        visible = state.visible()
        if not visible:
            return None
        cursor = state.cursor
        if cursor < 0 or cursor >= len(visible):
            return None
        return visible[cursor]

    @kb.add("up")
    def _(event: Any) -> None:
        state.move_up()
        event.app.invalidate()

    @kb.add("down")
    def _(event: Any) -> None:
        state.move_down()
        event.app.invalidate()

    @kb.add("c-p")
    def _(event: Any) -> None:
        # Emacs / claude-code parity: Ctrl+P moves up.
        state.move_up()
        event.app.invalidate()

    @kb.add("c-n")
    def _(event: Any) -> None:
        # Emacs / claude-code parity: Ctrl+N moves down.
        state.move_down()
        event.app.invalidate()

    @kb.add("pageup")
    def _(event: Any) -> None:
        state.page_up()
        event.app.invalidate()

    @kb.add("pagedown")
    def _(event: Any) -> None:
        state.page_down()
        event.app.invalidate()

    @kb.add("home")
    def _(event: Any) -> None:
        state.home()
        event.app.invalidate()

    @kb.add("end")
    def _(event: Any) -> None:
        state.end()
        event.app.invalidate()

    @kb.add("enter")
    def _(event: Any) -> None:
        item = _selected_item()
        event.app.exit(result=PickerResult(item=item))

    @kb.add("escape", eager=True)
    def _(event: Any) -> None:
        event.app.exit(result=PickerResult(item=None))

    @kb.add("c-c")
    def _(event: Any) -> None:
        # Treat Ctrl+C as cancel (claude-code parity). Eager-equivalent
        # so the binding wins over any default pt handler.
        event.app.exit(result=PickerResult(item=None))

    return Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        # Mouse off — wrong UX for a modal picker (user expects keyboard-
        # only) and pt's mouse handling adds latency on remote terminals.
        mouse_support=False,
        # erase_when_done=True wipes the picker's footprint after exit,
        # leaving the user's REPL surface intact. Without this the picker
        # would remain rendered above the next prompt.
        erase_when_done=True,
    )


async def run_picker(
    items: Sequence[PickerItem],
    *,
    title: str,
    initial_filter: str = "",
    page_size: int = DEFAULT_PAGE_SIZE,
    enable_filter: bool = True,
    total_count: int | None = None,
    style: Style | None = None,
) -> PickerResult:
    """Show a modal picker, return the user's selection (or cancel).

    Parameters:

    - ``items`` — the rows to display. Order is preserved as-is; the
      picker doesn't sort. An empty sequence short-circuits to
      ``PickerResult(item=None)`` without ever drawing.
    - ``title`` — single-line header (e.g. "Resume session").
    - ``initial_filter`` — pre-fill the filter buffer (lets a caller
      preserve filter state across re-invocations).
    - ``page_size`` — visible row count. Caller-tunable so tight panels
      can shrink and tall terminals can grow.
    - ``enable_filter`` — when False, the filter input row is hidden
      and the picker becomes a pure scroll list.
    - ``total_count`` — when the caller pre-truncated ``items`` (e.g.
      took the top-50 sessions out of 217 on disk) pass the underlying
      total here so the footer can render "showing 50 of 217 — type to
      filter".  Defaults to ``len(items)`` (no truncation hint).
    - ``style`` — optional pt :class:`Style` whose rules override the
      built-in defaults. Callers from the slash menu pass a tighter
      palette to match the rest of the REPL.

    Returns a :class:`PickerResult`; the ``item`` attribute is ``None``
    when the user cancelled.
    """
    if not items:
        return PickerResult(item=None)

    state = _PickerState(
        items=items, page_size=page_size, initial_filter=initial_filter,
    )
    state.clamp_after_filter_change()

    merged_style = _DEFAULT_STYLE
    if style is not None:
        # Merge: caller's style wins on conflicts. pt has no public
        # "merge" so we concatenate the rules (later rules win in pt's
        # cascade, matching CSS semantics).
        merged_style = Style(_DEFAULT_STYLE.style_rules + style.style_rules)

    app = _build_application(
        state,
        title=title,
        enable_filter=enable_filter,
        total_count=total_count if total_count is not None else len(items),
        style=merged_style,
    )
    result: PickerResult = await app.run_async()
    return result
