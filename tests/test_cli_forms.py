"""Tests for cli.forms.widget — unified form widget.

Tests the data path (state + answer assembly) by stubbing pt's
``Application.run_async`` with a coroutine that drives the keybinding
handlers manually. No live terminal is touched.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from prompt_toolkit.formatted_text import FormattedText, fragment_list_to_text
from prompt_toolkit.layout.containers import HSplit, VSplit

from cli.forms import FormCancelled, render_form
from cli.forms.widget import (
    FormOption,
    FormQuestion,
    _build_layout,
    _FormState,
    _has_any_preview,
    _QuestionState,
    _render_footer,
    _render_header,
    _render_options,
    _render_preview,
    _render_text_input,
)


class _FakeApp:
    """Captures the Application instance so tests can drive its handlers."""

    def __init__(self) -> None:
        self._exited = False

    def exit(self) -> None:
        self._exited = True

    def invalidate(self) -> None:
        pass


def _install_driver(
    monkeypatch: pytest.MonkeyPatch,
    driver: Callable[[Any], Awaitable[None]],
) -> None:
    """Patch ``Application.run_async`` to invoke a scripted driver."""
    from prompt_toolkit.application import Application

    async def _run_async(self: Application[Any]) -> None:
        await driver(self)

    monkeypatch.setattr(Application, "run_async", _run_async)


# Map human-readable key names to the canonical strings pt stores.
_KEY_ALIASES = {"enter": "c-m"}


def _press(app: Any, keys: str) -> None:
    """Find the first keybinding matching ``keys`` and invoke its handler."""
    target = _KEY_ALIASES.get(keys, keys)
    for binding in app.key_bindings.bindings:
        binding_keys = tuple(
            str(k.value) if hasattr(k, "value") else k for k in binding.keys
        )
        if binding_keys != (target,):
            continue
        # Skip bindings whose filter excludes the current mode (e.g. text-only
        # handlers when we're in options mode).
        if not binding.filter():
            continue
        binding.handler(_FakeEvent(app))
        return
    raise KeyError(f"no binding for {keys!r} (target={target!r})")


class _FakeEvent:
    def __init__(self, app: Any) -> None:
        # Hand the keybindings a fake app so calls to ``app.exit`` /
        # ``app.invalidate`` don't touch the real (not-running) pt loop.
        self.app = _FakeApp() if not isinstance(app, _FakeApp) else app
        self.data = ""


async def test_single_select_returns_chosen_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions: list[FormQuestion] = [
        {
            "question": "Pick a color",
            "header": "color",
            "multi_select": False,
            "options": [
                {"label": "red", "description": "warm"},
                {"label": "green", "description": "cool"},
            ],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, "down")  # cursor → green
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Pick a color": "green"}


async def test_cancel_via_escape_raises_form_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions: list[FormQuestion] = [
        {
            "question": "q?",
            "header": "q",
            "multi_select": False,
            "options": [{"label": "a", "description": ""}],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, "escape")

    _install_driver(monkeypatch, driver)
    with pytest.raises(FormCancelled):
        await render_form(questions)


async def test_multi_select_joins_labels_with_comma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions: list[FormQuestion] = [
        {
            "question": "Pick tags",
            "header": "tags",
            "multi_select": True,
            "options": [
                {"label": "alpha", "description": ""},
                {"label": "beta", "description": ""},
                {"label": "gamma", "description": ""},
            ],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, " ")       # check alpha (cursor 0)
        _press(app, "down")    # cursor → beta
        _press(app, "down")    # cursor → gamma
        _press(app, " ")       # check gamma
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Pick tags": "alpha, gamma"}


async def test_free_text_returns_buffer_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions: list[FormQuestion] = [
        {
            "question": "Your name?",
            "header": "name",
            "multi_select": False,
            "options": None,
            "free_text_placeholder": "type here",
        },
    ]

    async def driver(app: Any) -> None:
        # Drive the <any> handler to type characters.
        bindings = app.key_bindings.bindings
        any_handler = None
        for b in bindings:
            keys = tuple(str(k.value) if hasattr(k, "value") else k for k in b.keys)
            if keys == ("<any>",) and b.filter():
                any_handler = b.handler
                break
        assert any_handler is not None
        for ch in "Ada":
            event = _FakeEvent(app)
            event.data = ch
            any_handler(event)
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Your name?": "Ada"}


async def test_two_question_form_collects_both_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions: list[FormQuestion] = [
        {
            "question": "color?",
            "header": "c",
            "multi_select": False,
            "options": [
                {"label": "red", "description": ""},
                {"label": "blue", "description": ""},
            ],
            "free_text_placeholder": None,
        },
        {
            "question": "size?",
            "header": "s",
            "multi_select": False,
            "options": [
                {"label": "S", "description": ""},
                {"label": "L", "description": ""},
            ],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, "enter")        # commit q1 (red)
        _press(app, "down")         # cursor → L for q2
        _press(app, "enter")        # commit q2

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"color?": "red", "size?": "L"}


def _state(
    question: FormQuestion, qstate: _QuestionState | None = None,
) -> _FormState:
    return _FormState(
        questions=[question],
        per_question=[qstate or _QuestionState()],
    )


def _text(ft: FormattedText) -> str:
    return fragment_list_to_text(ft)


# --- key-binding driver scenarios (handler branches) ----------------------


async def test_up_arrow_wraps_cursor_to_last_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Up at the top must wrap to the last option so menus are circular."""
    questions: list[FormQuestion] = [
        {
            "question": "Pick",
            "header": "p",
            "multi_select": False,
            "options": [
                {"label": "first", "description": ""},
                {"label": "second", "description": ""},
                {"label": "third", "description": ""},
            ],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, "up")  # cursor 0 → wraps to last (third)
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Pick": "third"}


async def test_space_toggle_then_untoggle_clears_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotency: pressing Space twice on one row leaves it unchecked."""
    questions: list[FormQuestion] = [
        {
            "question": "Tags",
            "header": "t",
            "multi_select": True,
            "options": [
                {"label": "x", "description": ""},
                {"label": "y", "description": ""},
            ],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, " ")     # check x
        _press(app, " ")     # discard x (exercises the .discard branch)
        _press(app, "down")  # cursor → y
        _press(app, " ")     # check y
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Tags": "y"}


async def test_space_on_single_select_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Space must do nothing for single-select; Enter still picks the cursor."""
    questions: list[FormQuestion] = [
        {
            "question": "One",
            "header": "o",
            "multi_select": False,
            "options": [
                {"label": "only-a", "description": ""},
                {"label": "only-b", "description": ""},
            ],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, " ")     # early-return: no toggle in single-select
        _press(app, "down")  # cursor → only-b
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"One": "only-b"}


async def test_backspace_deletes_last_char_in_free_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backspace must trim the free-text buffer one char at a time."""
    questions: list[FormQuestion] = [
        {
            "question": "Word?",
            "header": "w",
            "multi_select": False,
            "options": None,
            "free_text_placeholder": "type",
        },
    ]

    async def driver(app: Any) -> None:
        bindings = app.key_bindings.bindings
        any_handler = None
        for b in bindings:
            keys = tuple(
                str(k.value) if hasattr(k, "value") else k for k in b.keys
            )
            if keys == ("<any>",) and b.filter():
                any_handler = b.handler
                break
        assert any_handler is not None
        for ch in "Cat":
            event = _FakeEvent(app)
            event.data = ch
            any_handler(event)
        _press(app, "c-h")  # backspace canonicalizes to c-h → "Ca"
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Word?": "Ca"}


async def test_backspace_on_empty_buffer_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backspace on an empty buffer must not underflow; answer stays empty."""
    questions: list[FormQuestion] = [
        {
            "question": "Maybe?",
            "header": "m",
            "multi_select": False,
            "options": None,
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, "c-h")   # backspace on empty buffer: guarded no-op
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Maybe?": ""}


async def test_ctrl_c_cancels_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C must abort the form exactly like Escape (no partial answers)."""
    questions: list[FormQuestion] = [
        {
            "question": "q?",
            "header": "q",
            "multi_select": False,
            "options": [{"label": "a", "description": ""}],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        _press(app, "c-c")

    _install_driver(monkeypatch, driver)
    with pytest.raises(FormCancelled):
        await render_form(questions)


async def test_run_async_no_exit_is_treated_as_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loop that returns without finishing must be reported as cancelled."""
    questions: list[FormQuestion] = [
        {
            "question": "q?",
            "header": "q",
            "multi_select": False,
            "options": [{"label": "a", "description": ""}],
            "free_text_placeholder": None,
        },
    ]

    async def driver(app: Any) -> None:
        return  # never presses Enter → state.finished stays False

    _install_driver(monkeypatch, driver)
    with pytest.raises(FormCancelled):
        await render_form(questions)


# --- side_by_side preview layout ------------------------------------------


async def test_preview_option_drives_side_by_side_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 'preview' option must select the two-column layout and still answer."""
    questions: list[FormQuestion] = [
        {
            "question": "Theme",
            "header": "theme",
            "multi_select": False,
            "options": [
                {"label": "light", "preview": "white bg"},
                {"label": "dark", "preview": "black bg"},
            ],
            "free_text_placeholder": None,
        },
    ]
    assert _has_any_preview(questions) is True

    async def driver(app: Any) -> None:
        _press(app, "down")  # cursor → dark
        _press(app, "enter")

    _install_driver(monkeypatch, driver)
    result = await render_form(questions)
    assert result == {"Theme": "dark"}


def test_build_layout_side_by_side_uses_vsplit_body() -> None:
    """side_by_side=True must place a VSplit body; False keeps a single HSplit."""
    question: FormQuestion = {
        "question": "q",
        "header": "h",
        "multi_select": False,
        "options": [{"label": "a", "preview": "P"}],
        "free_text_placeholder": None,
    }
    state = _state(question, _QuestionState(cursor=0))

    sbs_body = _build_layout(state, side_by_side=True).container.get_children()[1]
    plain_body = _build_layout(state, side_by_side=False).container.get_children()[1]

    assert isinstance(sbs_body, VSplit)
    assert isinstance(plain_body, HSplit)


def test_has_any_preview_ignores_empty_and_missing_previews() -> None:
    """Only a truthy preview string flips side_by_side; '' / missing do not."""
    no_preview: list[FormQuestion] = [
        {
            "question": "q",
            "header": "h",
            "multi_select": False,
            "options": [
                {"label": "a"},
                {"label": "b", "preview": None},
            ],
            "free_text_placeholder": None,
        },
        {
            "question": "free",
            "header": "f",
            "multi_select": False,
            "options": None,
            "free_text_placeholder": "x",
        },
    ]
    assert _has_any_preview(no_preview) is False


# --- pure render helpers (FormattedText fragment content) -----------------


def test_render_options_marks_checked_cursor_and_descriptions() -> None:
    """Multi-select rows must show [x] for checked, ❯ for cursor, and descs."""
    question: FormQuestion = {
        "question": "Pick",
        "header": "h",
        "multi_select": True,
        "options": [
            {"label": "alpha", "description": "first"},
            {"label": "beta", "description": ""},
        ],
        "free_text_placeholder": None,
    }
    state = _state(question, _QuestionState(cursor=1, checked={0}))
    rendered = _text(_render_options(state))
    assert "[x] alpha" in rendered
    assert "first" in rendered          # description line emitted
    assert "❯ [ ] beta" in rendered     # cursor glyph + unchecked mark


def test_render_header_shows_chip_and_progress() -> None:
    """Header must surface the header chip and a 1-based progress counter."""
    question: FormQuestion = {
        "question": "What now?",
        "header": "intro",
        "multi_select": False,
        "options": [{"label": "a", "description": ""}],
        "free_text_placeholder": None,
    }
    state = _FormState(
        questions=[question, question],
        per_question=[_QuestionState(), _QuestionState()],
        index=1,
    )
    rendered = _text(_render_header(state))
    assert "intro" in rendered
    assert "2/2" in rendered
    assert "What now?" in rendered


@pytest.mark.parametrize(
    ("multi_select", "is_free_text", "expected"),
    [
        (False, False, "Enter select"),
        (True, False, "Space toggle"),
        (False, True, "Enter confirm"),
    ],
)
def test_render_footer_hint_matches_mode(
    multi_select: bool, is_free_text: bool, expected: str,
) -> None:
    """Footer hint must match the active mode (single / multi / free-text)."""
    question: FormQuestion = {
        "question": "q",
        "header": "h",
        "multi_select": multi_select,
        "options": None if is_free_text else [{"label": "a", "description": ""}],
        "free_text_placeholder": None,
    }
    assert expected in _text(_render_footer(_state(question)))


@pytest.mark.parametrize(
    ("buffer", "placeholder", "needle"),
    [
        ("typed", "ph", "typed▎"),   # buffer wins over placeholder
        ("", "ph", "ph▎"),           # placeholder shown when empty
        ("", None, " ❯ ▎"),          # bare caret when nothing to show
    ],
)
def test_render_text_input_branches(
    buffer: str, placeholder: str | None, needle: str,
) -> None:
    """Free-text caret must reflect buffer, then placeholder, then bare state."""
    question: FormQuestion = {
        "question": "q",
        "header": "h",
        "multi_select": False,
        "options": None,
        "free_text_placeholder": placeholder,
    }
    state = _state(question, _QuestionState(text_buffer=buffer))
    assert needle in _text(_render_text_input(state))


_PREVIEW_SHOWN: list[FormOption] = [{"label": "a", "preview": "SHOWN"}]
_PREVIEW_MISSING: list[FormOption] = [{"label": "a"}]
_PREVIEW_OOB: list[FormOption] = [{"label": "a", "preview": "x"}]
_PREVIEW_NONE: list[FormOption] = []


@pytest.mark.parametrize(
    ("options", "cursor", "expected"),
    [
        (_PREVIEW_SHOWN, 0, "SHOWN"),
        (_PREVIEW_MISSING, 0, "(no preview)"),  # option lacks preview key
        (_PREVIEW_OOB, 9, ""),                  # cursor out of range
        (_PREVIEW_NONE, 0, ""),                 # no options at all
    ],
)
def test_render_preview_branches(
    options: list[FormOption], cursor: int, expected: str,
) -> None:
    """Preview pane: real text, (no preview), or empty for OOB/empty options."""
    question: FormQuestion = {
        "question": "q",
        "header": "h",
        "multi_select": False,
        "options": list(options),
        "free_text_placeholder": None,
    }
    rendered = _text(_render_preview(_state(question, _QuestionState(cursor=cursor))))
    assert rendered == expected if expected == "" else expected in rendered


def test_render_preview_empty_for_free_text_question() -> None:
    """Free-text questions have no preview pane regardless of cursor."""
    question: FormQuestion = {
        "question": "q",
        "header": "h",
        "multi_select": False,
        "options": None,
        "free_text_placeholder": "x",
    }
    assert _text(_render_preview(_state(question))) == ""
