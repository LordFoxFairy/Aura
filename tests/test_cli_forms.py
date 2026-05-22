"""Tests for cli.forms.widget — unified form widget.

Tests the data path (state + answer assembly) by stubbing pt's
``Application.run_async`` with a coroutine that drives the keybinding
handlers manually. No live terminal is touched.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from cli.forms import FormCancelled, render_form
from cli.forms.widget import FormQuestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
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
