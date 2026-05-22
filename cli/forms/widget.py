"""Unified form widget — render 1..4 questions in a single pt Application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from cli._coordination import prompt_mutex


class FormOption(TypedDict, total=False):
    label: str
    description: str
    preview: str | None


class FormQuestion(TypedDict, total=False):
    question: str
    header: str
    multi_select: bool
    options: list[FormOption] | None
    free_text_placeholder: str | None


class FormCancelled(Exception):
    """Raised when the user cancels the form via Esc / Ctrl-C."""


@dataclass
class _QuestionState:
    cursor: int = 0
    checked: set[int] = field(default_factory=set)
    text_buffer: str = ""


@dataclass
class _FormState:
    questions: list[FormQuestion]
    per_question: list[_QuestionState]
    index: int = 0
    cancelled: bool = False
    finished: bool = False

    @property
    def current_q(self) -> FormQuestion:
        return self.questions[self.index]

    @property
    def current_state(self) -> _QuestionState:
        return self.per_question[self.index]

    def current_options(self) -> list[FormOption]:
        opts = self.current_q.get("options")
        return opts or []

    def is_free_text(self) -> bool:
        return self.current_q.get("options") is None


def _has_any_preview(questions: list[FormQuestion]) -> bool:
    for q in questions:
        for opt in q.get("options") or []:
            if opt.get("preview"):
                return True
    return False


def _render_header(state: _FormState) -> FormattedText:
    q = state.current_q
    chip = q.get("header", "")
    progress = f"{state.index + 1}/{len(state.questions)}"
    frags: list[tuple[str, str]] = [
        ("class:form.chip", f" {chip} "),
        ("", "  "),
        ("class:form.progress", progress),
        ("", "\n\n"),
        ("bold", q.get("question", "")),
        ("", "\n"),
    ]
    return FormattedText(frags)


def _render_options(state: _FormState) -> FormattedText:
    qstate = state.current_state
    multi = state.current_q.get("multi_select", False)
    frags: list[tuple[str, str]] = []
    for i, opt in enumerate(state.current_options()):
        is_cursor = i == qstate.cursor
        mark = " "
        if multi:
            mark = "x" if i in qstate.checked else " "
        cursor_glyph = "❯" if is_cursor else " "
        style = "class:form.option.selected" if is_cursor else "class:form.option"
        line = f" {cursor_glyph} [{mark}] {opt.get('label', '')}\n"
        frags.append((style, line))
        desc = opt.get("description", "")
        if desc:
            desc_style = (
                "class:form.desc.selected" if is_cursor else "class:form.desc"
            )
            frags.append((desc_style, f"       {desc}\n"))
    return FormattedText(frags)


def _render_preview(state: _FormState) -> FormattedText:
    if state.is_free_text():
        return FormattedText([])
    opts = state.current_options()
    qstate = state.current_state
    if not opts or qstate.cursor >= len(opts):
        return FormattedText([])
    preview = opts[qstate.cursor].get("preview")
    if not preview:
        return FormattedText([("class:form.preview.empty", "  (no preview)\n")])
    return FormattedText([("class:form.preview", preview)])


def _render_text_input(state: _FormState) -> FormattedText:
    q = state.current_q
    placeholder = q.get("free_text_placeholder") or ""
    buf = state.current_state.text_buffer
    if buf:
        return FormattedText([("class:form.text", f" ❯ {buf}▎\n")])
    if placeholder:
        return FormattedText([
            ("class:form.text", " ❯ "),
            ("class:form.placeholder", f"{placeholder}▎\n"),
        ])
    return FormattedText([("class:form.text", " ❯ ▎\n")])


def _render_footer(state: _FormState) -> FormattedText:
    q = state.current_q
    if state.is_free_text():
        hint = " Enter confirm · Esc cancel"
    elif q.get("multi_select"):
        hint = " ↑/↓ move · Space toggle · Enter confirm · Esc cancel"
    else:
        hint = " ↑/↓ move · Enter select · Esc cancel"
    return FormattedText([("class:form.footer", hint + "\n")])


def _commit_current(state: _FormState, answers: dict[str, str]) -> None:
    q = state.current_q
    qstate = state.current_state
    question_text = q.get("question", "")
    if state.is_free_text():
        answers[question_text] = qstate.text_buffer
    elif q.get("multi_select"):
        labels = [
            (state.current_options()[i].get("label", ""))
            for i in sorted(qstate.checked)
        ]
        answers[question_text] = ", ".join(labels)
    else:
        opts = state.current_options()
        answers[question_text] = opts[qstate.cursor].get("label", "")


def _build_keybindings(
    state: _FormState, answers: dict[str, str],
) -> KeyBindings:
    kb = KeyBindings()
    in_options = Condition(lambda: not state.is_free_text())
    in_text = Condition(lambda: state.is_free_text())

    @kb.add("up", filter=in_options)
    def _(event: Any) -> None:
        n = len(state.current_options())
        if n:
            state.current_state.cursor = (state.current_state.cursor - 1) % n
            event.app.invalidate()

    @kb.add("down", filter=in_options)
    def _(event: Any) -> None:
        n = len(state.current_options())
        if n:
            state.current_state.cursor = (state.current_state.cursor + 1) % n
            event.app.invalidate()

    @kb.add(" ", filter=in_options)
    def _(event: Any) -> None:
        if not state.current_q.get("multi_select"):
            return
        cur = state.current_state.cursor
        if cur in state.current_state.checked:
            state.current_state.checked.discard(cur)
        else:
            state.current_state.checked.add(cur)
        event.app.invalidate()

    @kb.add("enter")
    @kb.add("c-m")
    @kb.add("c-j")
    def _(event: Any) -> None:
        _commit_current(state, answers)
        if state.index + 1 >= len(state.questions):
            state.finished = True
            event.app.exit()
            return
        state.index += 1
        event.app.invalidate()

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _(event: Any) -> None:
        state.cancelled = True
        event.app.exit()

    @kb.add("backspace", filter=in_text)
    def _(event: Any) -> None:
        buf = state.current_state.text_buffer
        if buf:
            state.current_state.text_buffer = buf[:-1]
            event.app.invalidate()

    @kb.add("<any>", filter=in_text)
    def _(event: Any) -> None:
        data = event.data
        if data and data.isprintable():
            state.current_state.text_buffer += data
            event.app.invalidate()

    return kb


def _build_layout(state: _FormState, side_by_side: bool) -> Layout:
    header_window = Window(
        FormattedTextControl(lambda: _render_header(state)),
        dont_extend_height=True,
    )
    options_window = ConditionalContainer(
        Window(
            FormattedTextControl(lambda: _render_options(state)),
            dont_extend_height=True,
        ),
        filter=Condition(lambda: not state.is_free_text()),
    )
    text_window = ConditionalContainer(
        Window(
            FormattedTextControl(lambda: _render_text_input(state)),
            dont_extend_height=True,
        ),
        filter=Condition(state.is_free_text),
    )
    footer_window = Window(
        FormattedTextControl(lambda: _render_footer(state)),
        height=1, dont_extend_height=True,
    )

    body: Any
    if side_by_side:
        preview_window = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: _render_preview(state)),
                dont_extend_height=True,
            ),
            filter=Condition(lambda: not state.is_free_text()),
        )
        body = VSplit([
            HSplit([options_window, text_window]),
            Window(width=Dimension.exact(2)),
            HSplit([preview_window]),
        ])
    else:
        body = HSplit([options_window, text_window])

    return Layout(HSplit([header_window, body, footer_window]))


async def render_form(questions: list[FormQuestion]) -> dict[str, str]:
    """Render 1..4 questions via prompt_toolkit; return {question_text: answer}.

    Layout: a single pt ``Application`` advances through questions
    one-at-a-time, mutating ``_FormState.index`` on Enter. The preview
    pane is laid out side-by-side when any option declares a preview;
    otherwise the layout is vertical only.
    """
    state = _FormState(
        questions=list(questions),
        per_question=[_QuestionState() for _ in questions],
    )
    answers: dict[str, str] = {}
    kb = _build_keybindings(state, answers)
    layout = _build_layout(state, side_by_side=_has_any_preview(state.questions))

    app: Application[Any] = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )

    async with prompt_mutex():
        await app.run_async()

    if state.cancelled or not state.finished:
        raise FormCancelled
    return answers
