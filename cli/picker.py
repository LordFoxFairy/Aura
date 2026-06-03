"""Scrollable, filterable item picker for the REPL."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, VSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

DEFAULT_PAGE_SIZE: int = 8

_CURSOR_GLYPH: str = "❯"
_CURSOR_COL_WIDTH: int = get_cwidth(_CURSOR_GLYPH) + 1

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
    # Property form so frozen dataclasses satisfy the protocol under mypy.
    @property
    def label(self) -> str: ...

    @property
    def sublabel(self) -> str | None: ...

    @property
    def value(self) -> object: ...


@dataclass(frozen=True)
class SimplePickerItem:
    label: str
    sublabel: str | None
    value: object


@dataclass(frozen=True)
class PickerResult:
    item: PickerItem | None


def _terminal_width(default: int = 80) -> int:
    try:
        return max(40, min(200, shutil.get_terminal_size((default, 24)).columns))
    except OSError:
        return default


def _truncate_to_width(text: str, max_cols: int) -> str:
    if max_cols <= 0:
        return ""
    if get_cwidth(text) <= max_cols:
        return text
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
    if not needle:
        return True
    lo = needle.lower()
    if lo in item.label.lower():
        return True
    sub = item.sublabel or ""
    return lo in sub.lower()


class _PickerState:
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
        return [it for it in self.items if _matches_filter(it, self.filter_text)]

    def clamp_after_filter_change(self) -> None:
        n = len(self.visible())
        if n == 0:
            self.cursor = 0
            self.viewport_top = 0
            return
        if self.cursor >= n:
            self.cursor = n - 1
        if self.cursor < self.viewport_top:
            self.viewport_top = self.cursor
        if self.cursor >= self.viewport_top + self.page_size:
            self.viewport_top = self.cursor - self.page_size + 1
        self.viewport_top = max(0, min(self.viewport_top, max(0, n - self.page_size)))

    def move_down(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        self.cursor = (self.cursor + 1) % n
        if self.cursor == 0:
            self.viewport_top = 0
            return
        if self.cursor >= self.viewport_top + self.page_size:
            self.viewport_top = self.cursor - self.page_size + 1

    def move_up(self) -> None:
        n = len(self.visible())
        if n == 0:
            return
        if self.cursor == 0:
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
    visible = state.visible()
    if not visible:
        return FormattedText([("class:picker.empty", "  (no matches)\n")])

    fragments: list[tuple[str, str]] = []
    width = _terminal_width()
    top = max(0, min(state.viewport_top, max(0, len(visible) - state.page_size)))
    bottom = min(len(visible), top + state.page_size)

    if top > 0:
        fragments.append(("class:picker.footer", f"  ({top} more above)\n"))

    for idx in range(top, bottom):
        item = visible[idx]
        is_cursor = idx == state.cursor

        sublabel = item.sublabel or ""
        sublabel_w = get_cwidth(sublabel) if sublabel else 0
        gap = 1 if sublabel else 0
        label_budget = max(1, width - _CURSOR_COL_WIDTH - sublabel_w - gap - 1)
        label = _truncate_to_width(item.label, label_budget)

        row_style = "class:picker.selected" if is_cursor else ""

        if is_cursor:
            fragments.append(("class:picker.cursor", _CURSOR_GLYPH + " "))
        else:
            fragments.append((row_style, " " * _CURSOR_COL_WIDTH))

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

    remaining = len(visible) - bottom
    if remaining > 0:
        fragments.append(("class:picker.footer", f"  ({remaining} more below)\n"))

    return FormattedText(fragments)


def _build_application(
    state: _PickerState,
    *,
    title: str,
    enable_filter: bool,
    total_count: int,
    style: Style,
) -> Application[PickerResult]:
    filter_buffer = Buffer(multiline=False)
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
        if state.filter_text:
            parts.append((
                "class:picker.footer.match",
                f"  {visible_n} matching of {total_count}",
            ))
        elif total_count > len(state.items):
            parts.append((
                "class:picker.footer",
                f"  showing {len(state.items)} of {total_count} — type to filter",
            ))
        else:
            parts.append(("class:picker.footer", f"  {visible_n} items"))
        parts.append((
            "class:picker.footer",
            "    ↑/↓ navigate · Enter select · Esc cancel",
        ))
        parts.append(("", "\n"))
        return FormattedText(parts)

    title_window = Window(
        FormattedTextControl(_title_text), height=1, dont_extend_height=True,
    )
    items_window = Window(
        FormattedTextControl(_items_text),
        height=Dimension.exact(state.page_size + 2),
        dont_extend_height=True,
    )
    footer_window = Window(
        FormattedTextControl(_footer_text),
        height=1,
        dont_extend_height=True,
    )

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
        filter=bool(enable_filter),
    )

    layout = Layout(
        HSplit([title_window, items_window, filter_row, footer_window]),
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

    def _bind_move(key: str, action: Callable[[], None]) -> None:
        @kb.add(key)
        def _(event: Any) -> None:
            action()
            event.app.invalidate()

    _bind_move("up", state.move_up)
    _bind_move("down", state.move_down)
    _bind_move("c-p", state.move_up)
    _bind_move("c-n", state.move_down)
    _bind_move("pageup", state.page_up)
    _bind_move("pagedown", state.page_down)
    _bind_move("home", state.home)
    _bind_move("end", state.end)

    @kb.add("enter")
    def _(event: Any) -> None:
        event.app.exit(result=PickerResult(item=_selected_item()))

    @kb.add("escape", eager=True)
    def _(event: Any) -> None:
        event.app.exit(result=PickerResult(item=None))

    @kb.add("c-c")
    def _(event: Any) -> None:
        event.app.exit(result=PickerResult(item=None))

    return Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
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
    if not items:
        return PickerResult(item=None)

    state = _PickerState(
        items=items, page_size=page_size, initial_filter=initial_filter,
    )
    state.clamp_after_filter_change()

    merged_style = _DEFAULT_STYLE
    if style is not None:
        # pt has no public Style.merge; rule concatenation matches CSS cascade.
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
