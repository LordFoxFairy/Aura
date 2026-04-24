"""Async REPL — the main loop that drives the agent from user input."""

from __future__ import annotations

import asyncio
import contextlib
import random
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aura import __version__
from aura.cli.commands import build_default_registry, dispatch
from aura.cli.completion import SlashCommandCompleter, resolve_history_path
from aura.cli.render import Renderer
from aura.cli.spinner import ThinkingSpinner
from aura.core.agent import Agent
from aura.core.commands import CommandRegistry
from aura.core.persistence import journal
from aura.schemas.permissions import StatusLineConfig

InputFn = Callable[[str], Awaitable[str]]


#: Order used by the shift+tab mode-cycle keybinding. ``bypass`` is
#: deliberately absent — it's dangerous (allow-everything) and can only
#: be enabled via ``--bypass-permissions`` at CLI startup, never mid-session.
_MODE_CYCLE: tuple[str, ...] = ("default", "accept_edits", "plan")


#: Window during which a second Ctrl+C is treated as "confirm exit".
#: Mirrors claude-code's ``DOUBLE_PRESS_TIMEOUT_MS = 800`` in
#: ``src/hooks/useDoublePress.ts`` — fast enough that accidental double-
#: presses don't exit, slow enough that intentional double-taps succeed.
_CTRL_C_DOUBLE_PRESS_SECONDS: float = 0.8


class _CtrlCState:
    """Shared mutable state for the Ctrl+C double-press handler.

    Tracks the wall-clock time of the last bare Ctrl+C so the *second*
    press within :data:`_CTRL_C_DOUBLE_PRESS_SECONDS` can escalate to
    exit. Also stashes the last hint text so the bottom toolbar can
    render "Press Ctrl+C again to exit" while the window is open.

    Used as a module-private helper; callers do not construct it
    directly — :func:`_build_mode_key_bindings` allocates one per REPL
    session and shares it with :func:`_make_bottom_toolbar`.
    """

    __slots__ = ("last_press_at",)

    def __init__(self) -> None:
        # Seconds since epoch. 0.0 means "no prior press", so the first
        # comparison against ``now - last_press_at > window`` always
        # treats the initial press as "first".
        self.last_press_at: float = 0.0

    def hint_active(self, now: float) -> bool:
        """True iff the prior Ctrl+C landed within the double-press window."""
        return (
            self.last_press_at > 0.0
            and (now - self.last_press_at) <= _CTRL_C_DOUBLE_PRESS_SECONDS
        )


#: Startup tips rotated on each welcome banner render. Kept as a stable
#: module-level tuple so tests can assert membership without pulling in
#: the random pick. Only mention features that exist today — adding dead
#: tips teaches the user wrong reflexes. Order is not meaningful.
_STARTUP_TIPS: tuple[str, ...] = (
    "shift+tab cycles permission modes (default → accept_edits → plan)",
    "esc resets permission mode to default",
    "/help lists every slash command available right now",
    "/model switches models mid-session",
    "/compact trims history while keeping key context",
    "Ctrl+R searches your input history",
    "Ctrl+D exits cleanly; /exit does the same from any prompt",
    "AURA.md at project root sets persistent context",
    "--bypass-permissions unlocks allow-everything mode for one session",
    "--verbose prints per-turn token / model / latency summaries",
)


def _cycle_mode(current: str) -> str:
    """Advance ``current`` one step through :data:`_MODE_CYCLE`.

    If ``current`` is not in the cycle (e.g. ``"bypass"``), returns it
    unchanged — callers should check and short-circuit so the user sees
    a clear message instead of an unexpected mode flip.
    """
    if current not in _MODE_CYCLE:
        return current
    idx = _MODE_CYCLE.index(current)
    return _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]


def _build_mode_key_bindings(
    agent: Agent,
    console: Console,
    ctrl_c_state: _CtrlCState | None = None,
) -> KeyBindings:
    """Build a KeyBindings carrying Aura's prompt-level bindings.

    Mode-change feedback is **visual only** — ``event.app.invalidate()``
    redraws pt's ``bottom_toolbar`` whose ``mode:`` field reflects the
    new mode. Printing a confirmation line on every press spams the
    scrollback on rapid/repeated cycling (operator report + img_2.png:
    dozens of stacked ``mode: X (press shift+tab ...)`` lines).

    Bindings:

    - ``s-tab`` → cycle permission mode (default → accept_edits → plan).
      Under bypass: silently no-op (bypass is sticky; the startup banner
      already warns once).
    - ``escape`` → reset permission mode to default. **Non-eager** on
      purpose: eager escape would swallow the ESC prefix of meta-key
      sequences (Alt/Meta+Enter arrives as ESC then CR), breaking the
      multi-line input bindings below. The cost is a ~0.5s wait before
      a bare Esc resets the mode — acceptable UX.
    - ``escape, enter`` → insert literal ``\\n`` in the buffer. This is
      the universal Alt/Meta+Enter sequence: works on every terminal
      including macOS Terminal.app (Option-Enter) and iTerm2.
    - ``c-j`` → insert literal ``\\n`` in the buffer. Some terminals
      (iTerm2 custom keymap, many Linux terminals) let users remap
      Shift+Enter to send Ctrl+J; this binding makes that Just Work.
      Shift+Enter itself is NOT directly bindable — most terminals
      don't distinguish it from Enter by default.
    - ``c-c`` → three-state handler mirroring claude-code
      ``src/hooks/useExitOnCtrlCD.ts`` + ``useDoublePress.ts``:

      1. Buffer has text → clear it (no exit).
      2. Buffer is empty, first press → arm the "press again to exit"
         state; bottom toolbar hint is rendered for
         :data:`_CTRL_C_DOUBLE_PRESS_SECONDS`.
      3. Buffer is empty, second press within the window → raise
         ``KeyboardInterrupt`` via pt's ``app.exit(exception=...)`` so
         the outer REPL loop takes the exit path.

      Cancelling an in-flight turn is NOT this binding's concern —
      during a turn, pt's ``prompt_async`` is NOT running; Ctrl+C goes
      to Python's default SIGINT handler and reaches
      :func:`_run_turn`'s inner try/except which cancels the stream
      task cleanly. See the ``_run_turn`` docstring.

    Plain Enter keeps its pt default "accept-line" (submit). We
    deliberately do NOT set ``multiline=True`` on the session because
    that would invert Enter's meaning and force a non-standard submit
    key (Esc-Enter / Ctrl+D).

    ``console`` is retained in the signature (not currently used in any
    binding) so future bindings that legitimately need to print out-of-band
    can reuse the same wiring without another constructor churn.

    ``ctrl_c_state`` is shared with :func:`_make_bottom_toolbar` so the
    toolbar renders a live "Press Ctrl+C again to exit" hint during the
    double-press window. A ``None`` state (the common test path) falls
    back to allocating a fresh per-binding state — the hint still works
    inside pt's own surface; only cross-surface toolbar visibility
    requires sharing.
    """
    kb = KeyBindings()
    del console  # reserved in signature for future bindings; see docstring.
    state = ctrl_c_state if ctrl_c_state is not None else _CtrlCState()

    @kb.add("s-tab")
    def _(event: Any) -> None:
        current = agent.mode
        if current == "bypass":
            return
        agent.set_mode(_cycle_mode(current))
        event.app.invalidate()

    @kb.add("escape")
    def _(event: Any) -> None:
        # Non-eager so meta-key sequences (escape, enter) still match.
        # Bypass mode is sticky for the whole session by design.
        if agent.mode == "bypass" or agent.mode == "default":
            return
        agent.set_mode("default")
        event.app.invalidate()

    @kb.add("escape", "enter")
    def _(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-j")
    def _(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-c")
    def _(event: Any) -> None:
        """Claude-code-style Ctrl+C: clear / arm / exit (no data loss)."""
        buffer = event.current_buffer
        now = time.monotonic()
        if buffer.text:
            # Case 1: text present → discard it. Don't reset the
            # double-press timer — clearing input IS a deliberate
            # use of Ctrl+C, not a bid to exit.
            buffer.reset()
            event.app.invalidate()
            return
        if state.hint_active(now):
            # Case 3: second bare Ctrl+C within the window → EXIT.
            # Signalling via app.exit(exception=...) hands control back
            # to prompt_async which re-raises. Outer REPL loop catches
            # it and takes the clean-exit path.
            state.last_press_at = 0.0
            event.app.exit(exception=KeyboardInterrupt())
            return
        # Case 2: first bare Ctrl+C. Arm the double-press window and
        # invalidate so the bottom toolbar can paint the hint.
        state.last_press_at = now
        event.app.invalidate()

    return kb


def _build_prompt_session(
    registry: CommandRegistry,
    agent: Agent | None = None,
    last_turn_seconds_getter: Callable[[], float] | None = None,
    console: Console | None = None,
    statusline: StatusLineConfig | None = None,
) -> PromptSession[str]:
    """Construct a PromptSession wired with history, completion, and an
    informational bottom toolbar.

    - ``FileHistory`` at ``~/.aura/history`` → up-arrow cycles across sessions.
    - ``search_ignore_case=True`` → Ctrl+R reverse search, case-insensitive.
    - ``SlashCommandCompleter`` with a live registry getter → Skill / MCP
      commands registered after PromptSession construction still complete.
    - ``complete_while_typing=False`` → menu only on Tab, never while typing
      normal prompts (would be noisy).
    - ``bottom_toolbar`` is a **callable** closing over ``agent``; pt
      re-invokes it on every render so the numbers track live state
      without polling. Shows model / context-pressure bar / pinned-cache /
      mode / cwd. Keybinding hints live in the welcome banner (shown once)
      — the bar is for always-relevant stateful info.
    - ``style`` overrides pt's built-in ``reverse`` styling on
      ``bottom-toolbar`` so the footer blends with the terminal
      background instead of showing as a high-contrast inverted bar.
      That default (designed for full-screen TUIs) fights our rich
      welcome panel, which is plain-bg + dim text. Matching "looks
      like a dim line" keeps the surface consistent.

    ``agent=None`` disables the bottom_toolbar (keeps the function usable
    in tests that only exercise history / completion wiring).
    """
    history = FileHistory(str(resolve_history_path()))
    completer = SlashCommandCompleter(lambda: registry)
    # Ctrl+C double-press state is shared between the key binding and
    # the bottom-toolbar renderer so the "Press Ctrl+C again to exit"
    # hint is live-visible exactly while the window is armed.
    ctrl_c_state = _CtrlCState() if agent is not None else None
    bottom_toolbar = (
        _make_bottom_toolbar(
            agent, last_turn_seconds_getter, statusline, ctrl_c_state,
        )
        if agent is not None
        else None
    )
    style = Style.from_dict({
        "bottom-toolbar": "noreverse",
        "bottom-toolbar.text": "noreverse",
    })
    # Shift+Tab cycles permission mode at the prompt — attach the
    # binding only when we have both an agent (to mutate) and a console
    # (to print confirmation). Missing either falls back to pt's default
    # bindings (tests that only exercise history/completion won't trip).
    key_bindings = (
        _build_mode_key_bindings(agent, console, ctrl_c_state)
        if agent is not None and console is not None
        else None
    )
    return PromptSession(
        history=history,
        completer=completer,
        complete_while_typing=False,
        search_ignore_case=True,
        bottom_toolbar=bottom_toolbar,
        style=style,
        key_bindings=key_bindings,
    )


def _make_bottom_toolbar(
    agent: Agent,
    last_turn_seconds_getter: Callable[[], float] | None = None,
    statusline: StatusLineConfig | None = None,
    ctrl_c_state: _CtrlCState | None = None,
) -> Callable[[], Any]:
    """Build a pt-compatible bottom_toolbar callable that reads live agent
    state on each render. Closing over ``agent`` rather than snapshotting
    the values is the whole point: every turn's new ``_token_stats`` +
    any mid-session ``/model`` switch show up in the bar without manual
    re-wiring. ``Agent.mode`` and ``Agent.context_window`` encapsulate
    the mode / window resolution so the toolbar stays a thin projection.

    ``last_turn_seconds_getter`` is a closure over a REPL-scope mutable
    so the bar picks up the most recent turn's wall-clock duration without
    stashing it on agent state (it's display-only REPL info).

    When ``statusline`` is set and active, the callable runs the user's
    shell command to produce the bar text; each render kicks off an
    async refresh and returns the LAST cached value so pt (whose
    toolbar callable must return synchronously) never blocks on the
    subprocess. pt gets told to ``invalidate()`` when the refresh
    completes so the new text shows on the very next paint.
    """
    from prompt_toolkit.formatted_text import HTML

    from aura.cli import buddy as _buddy
    from aura.cli.status_bar import render_bottom_toolbar_html

    # Cache for hook-driven bars: holds the LAST successful hook render
    # (or ``None`` pre-first-fire) so pt always has SOMETHING to paint
    # synchronously. Hook runs async in the background; on completion
    # it rewrites this cell and invalidates the pt app.
    hook_cache: list[Any] = [None]
    refresh_in_flight: list[bool] = [False]

    def _buddy_suffix_html() -> str:
        """Return a pt-HTML fragment for the buddy, or empty string.

        The buddy module returns a plain ``"🦆 happy"`` string; we wrap
        it in ``<ansigray>`` so it matches the rest of the dim bar.
        Reads the ``ui.buddy_enabled`` flag off ``agent.config`` when
        that attribute exists — defaults to ``True`` otherwise so
        pre-config bare Agent tests still get the default-on behavior.
        Any failure (missing attribute, unexpected emoji in an HTML
        entity position, etc.) silently returns ``""`` — the buddy is
        eye candy, never load-bearing."""
        try:
            enabled = True
            cfg = getattr(agent, "config", None)
            if cfg is not None:
                ui = getattr(cfg, "ui", None)
                if ui is not None:
                    enabled = bool(getattr(ui, "buddy_enabled", True))
            frag = _buddy.buddy_status_fragment(
                state=agent.state, enabled=enabled,
            )
            if not frag:
                return ""
            return f" · <ansigray>{frag}</ansigray>"
        except Exception:  # noqa: BLE001 — display path must never crash
            return ""

    def _snapshot() -> dict[str, Any]:
        stats = agent.state.custom.get("_token_stats", {})
        return {
            "input_tokens": int(stats.get("last_input_tokens", 0)),
            "cache_tokens": int(stats.get("last_cache_read_tokens", 0)),
            "model": agent.current_model or "",
            "last_secs": (
                last_turn_seconds_getter() if last_turn_seconds_getter else 0.0
            ),
            "pinned": agent.pinned_tokens_estimate,
            "window": agent.context_window,
            "mode": agent.mode,
            "cwd": Path.cwd(),
        }

    def _render_default(snap: dict[str, Any]) -> Any:
        base = render_bottom_toolbar_html(
            model=snap["model"] or None,
            input_tokens=snap["input_tokens"],
            cache_read_tokens=snap["cache_tokens"],
            pinned_estimate_tokens=snap["pinned"],
            context_window=snap["window"],
            mode=snap["mode"],
            cwd=snap["cwd"],
            last_turn_seconds=snap["last_secs"],
        )
        # Append buddy fragment AFTER all existing pieces so the normal
        # status bar is never visually disrupted; when the user opts out
        # (env var / config flag) the suffix is empty and we return the
        # original HTML untouched.
        suffix = _buddy_suffix_html()
        if not suffix:
            return base
        return HTML(base.value + suffix)

    hook_active = statusline is not None and statusline.is_active

    def _ctrl_c_hint_html() -> Any | None:
        """Return a pt ``HTML`` carrying the double-press hint, or None.

        When the Ctrl+C double-press window is armed, the hint
        REPLACES the normal status bar — claude-code shows the same
        pattern on its bottom slot (``PromptInputFooterLeftSide.tsx``
        line 150: ``Press {exitMessage.key} again to exit``). Using
        the toolbar slot keeps the hint inside pt's live surface
        (no scrollback pollution) and it vanishes automatically when
        the window elapses (pt re-renders and ``hint_active`` returns
        False).
        """
        if ctrl_c_state is None:
            return None
        if not ctrl_c_state.hint_active(time.monotonic()):
            return None
        from prompt_toolkit.formatted_text import HTML
        return HTML("<ansiyellow>Press Ctrl+C again to exit</ansiyellow>")

    def _render() -> Any:
        hint = _ctrl_c_hint_html()
        if hint is not None:
            return hint
        snap = _snapshot()
        if not hook_active or statusline is None:
            return _render_default(snap)
        # Hook path — return cached value immediately, refresh in bg.
        if not refresh_in_flight[0]:
            refresh_in_flight[0] = True
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_refresh(snap))
            except RuntimeError:
                # No running loop (e.g. smoke tests invoking the callable
                # outside an event loop); just synchronously fall back.
                refresh_in_flight[0] = False
                return _render_default(snap)
        cached = hook_cache[0]
        return cached if cached is not None else _render_default(snap)

    async def _refresh(snap: dict[str, Any]) -> None:
        """Background job: run the hook, update the cache, invalidate pt.

        Any exception is swallowed — the default render is already in the
        cache fallback path, so a hook crash is visually indistinguishable
        from the hook being unset. Errors go to the journal for debugging.
        """
        from aura.cli.status_bar import render_bottom_toolbar_with_hook

        try:
            assert statusline is not None
            result = await render_bottom_toolbar_with_hook(
                model=snap["model"] or None,
                input_tokens=snap["input_tokens"],
                cache_read_tokens=snap["cache_tokens"],
                pinned_estimate_tokens=snap["pinned"],
                context_window=snap["window"],
                mode=snap["mode"],
                cwd=snap["cwd"],
                last_turn_seconds=snap["last_secs"],
                hook_command=statusline.command,
                hook_timeout_s=statusline.timeout_ms / 1000.0,
            )
            hook_cache[0] = result
            # Ask pt to repaint — without this the new cached value would
            # only show on the NEXT user keystroke.
            try:
                from prompt_toolkit.application.current import (
                    get_app_or_none,
                )
                app = get_app_or_none()
                if app is not None:
                    app.invalidate()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001 — display-path resilience
            journal.write(
                "statusline_hook_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        finally:
            refresh_in_flight[0] = False

    return _render


def _make_prompt_session_input(session: PromptSession[str]) -> InputFn:
    async def _read(prompt: str) -> str:
        return await session.prompt_async(prompt)

    return _read


async def _default_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def run_repl_async(
    agent: Agent,
    *,
    input_fn: InputFn | None = None,
    console: Console | None = None,
    verbose: bool = False,
    bypass: bool = False,
    statusline: StatusLineConfig | None = None,
) -> None:
    journal.write("repl_started")
    _console = console if console is not None else Console()
    renderer = Renderer(_console)
    registry = build_default_registry(agent=agent)

    # Wall-clock duration of the most recent turn. Single-element list
    # (not a bare float) so both the toolbar closure and the main loop
    # can read / write the same cell without a dedicated class. Stays
    # in the REPL closure — deliberately NOT on agent state.
    last_turn_seconds: list[float] = [0.0]

    # Resolution order for the input function:
    # 1. Explicit ``input_fn`` override (tests / non-interactive callers) —
    #    always wins. Keeps the scripted test harness working verbatim.
    # 2. If stdin is a TTY, build a PromptSession (history, completion,
    #    Ctrl+R, bottom toolbar).
    # 3. Otherwise fall back to plain ``input()`` so piped / dumb terminals
    #    don't hang waiting on a prompt_toolkit renderer they can't drive.
    if input_fn is not None:
        _input: InputFn = input_fn
    elif sys.stdin.isatty():
        _input = _make_prompt_session_input(
            _build_prompt_session(
                registry,
                agent=agent,
                last_turn_seconds_getter=lambda: last_turn_seconds[0],
                console=_console,
                statusline=statusline,
            )
        )
    else:
        _input = _default_input

    _print_welcome(agent, _console)

    # In bypass mode the startup banner scrolls off after a few turns;
    # encode bypass into the prompt string so every line reminds the user
    # they're in "allow-everything" mode. ANSI red around the marker —
    # works in any terminal that supports colour; plain terminals just see
    # the escapes as text (ugly but not dangerous).
    prompt_str = "\x1b[31maura[!bypass]>\x1b[0m " if bypass else "aura> "

    while True:
        try:
            line = await _input(prompt_str)
        except (EOFError, KeyboardInterrupt):
            journal.write("repl_exit", reason="eof_or_ctrlc")
            _console.print()
            return

        # Empty / whitespace-only input: reprompt silently. Sending an empty
        # HumanMessage to the model always 400s (providers reject empty user
        # turns), so it's a pure UX nuisance to round-trip it.
        if not line.strip():
            continue

        journal.write("user_input", line=line[:500])

        result = await dispatch(line, agent, registry)
        if result.handled:
            journal.write(
                "slash_command",
                line=line[:200],
                kind=result.kind,
            )
            if result.kind == "exit":
                journal.write("repl_exit", reason="slash_exit")
                return
            if result.text:
                _console.print(result.text)
            continue

        try:
            last_turn_seconds[0] = await _run_turn(
                agent, line, renderer, _console,
            )
        except Exception as exc:  # noqa: BLE001 — REPL resilience
            # Don't catch BaseException: KeyboardInterrupt/SystemExit/
            # CancelledError must still propagate up to main() so the
            # whole process can exit cleanly. But a network hiccup, an
            # LLM client bug, or a provider 500 should NOT tear down the
            # user's interactive session — surface the error, journal it,
            # loop back to the prompt.
            journal.write(
                "turn_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            _console.print(
                f"[red]turn failed: {type(exc).__name__}: {exc}[/red]"
            )

        # Post-turn status checkpoint — printed AFTER every turn so the
        # operator still sees the model/tokens/mode/cwd summary while pt's
        # bottom toolbar is hidden during streaming. Runs regardless of
        # --verbose (verbose adds the cumulative-totals summary on top).
        _print_post_turn_status(agent, _console, last_turn_seconds[0])

        if verbose:
            _print_verbose_summary(agent, _console)


#: Frame set for the leading glyph on the welcome banner — same characters
#: the in-turn :class:`ThinkingSpinner` uses (see ``aura/cli/spinner.py``),
#: palindromic so the animation bounces rather than resets. Matches
#: claude-code ``src/components/Spinner/utils.ts::getDefaultCharacters``
#: (darwin set: ``['·', '✢', '✳', '✶', '✻', '✽']``). Using the SAME set
#: as the thinking spinner sells the continuity: welcome banner glyph at
#: startup, same-family glyph spinning during a turn.
_BANNER_SPINNER_FRAMES: tuple[str, ...] = (
    "·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢",
)
#: Total on-screen time for the welcome animation. Short enough that a
#: fast startup doesn't feel gated; long enough to signal "this is a
#: live UI" (~10 frames at 120ms = 1.2s).
_BANNER_ANIMATION_SECONDS: float = 1.2
_BANNER_FRAME_INTERVAL: float = 0.12
#: The glyph the banner SETTLES on after animation — the final ``✱``
#: matches the wordmark we've shipped since v0.1 and renders cleanly in
#: every terminal we support (braille glyphs like ``⏺`` are missing on
#: older Linux fonts).
_BANNER_SETTLE_GLYPH: str = "✱"


def _render_welcome_panel(agent: Agent, glyph: str) -> Panel:
    """Build the welcome Panel with the given leading glyph.

    Factored out of :func:`_print_welcome` so the startup animation can
    re-render the same panel with a different frame on each tick without
    duplicating the layout code.
    """
    cwd_display = str(Path.cwd())
    home = str(Path.home())
    if cwd_display == home or cwd_display.startswith(home + "/"):
        cwd_display = "~" + cwd_display[len(home):]

    tip = random.choice(_STARTUP_TIPS)

    body = Text()
    body.append(f"{glyph} Aura", style="bold")
    body.append(f" v{__version__}  ·  ", style="dim")
    body.append("/help", style="cyan")
    body.append("  ·  Ctrl+D to exit  ·  ", style="dim")
    body.append("shift+tab cycles mode\n", style="dim")
    body.append("model: ", style="dim")
    body.append(f"{agent.current_model}\n", style="")
    body.append("cwd:   ", style="dim")
    body.append(f"{cwd_display}\n", style="")
    body.append("tip:   ", style="dim")
    body.append(tip, style="dim")
    return Panel(body, border_style="cyan", padding=(0, 2), expand=False)


def _print_welcome(agent: Agent, console: Console) -> None:
    """Print the startup welcome panel.

    Visual parity with claude-code's ``REPL.tsx`` header spinner: the
    leading glyph animates through the spinner frame set for a brief
    window after launch (signalling "live UI"), then settles on a
    stable glyph — ``✱``, the Aura wordmark. During a turn the same
    family of glyphs drives :class:`ThinkingSpinner`, so the visual
    language carries through consistently.

    The animation uses ``rich.live.Live(transient=False)`` so the final
    frame stays in scrollback (tests + screenshots preserve the banner).
    Non-TTY callers (``console.is_terminal == False``) short-circuit the
    animation and print the settled banner directly — StringIO tests
    asserting ``"✱ Aura"`` in the captured output still pass.
    """
    # Tests and piped/dumb terminals don't render Live frames — drop
    # straight to the final banner. ``console.is_terminal`` is False
    # for StringIO-backed consoles (the test harness default).
    if not console.is_terminal:
        console.print(_render_welcome_panel(agent, _BANNER_SETTLE_GLYPH))
        return

    import time as _time

    from rich.live import Live

    total_frames = max(
        1, int(_BANNER_ANIMATION_SECONDS / _BANNER_FRAME_INTERVAL),
    )
    # Live.update with transient=False leaves the LAST rendered frame
    # in scrollback when stopped. The last frame we update to is the
    # settled ✱, so the final on-screen state exactly matches what
    # ``_render_welcome_panel(agent, "✱")`` would print.
    with Live(
        _render_welcome_panel(agent, _BANNER_SPINNER_FRAMES[0]),
        console=console,
        refresh_per_second=1 / _BANNER_FRAME_INTERVAL,
        transient=False,
    ) as live:
        for i in range(1, total_frames):
            _time.sleep(_BANNER_FRAME_INTERVAL)
            frame = _BANNER_SPINNER_FRAMES[i % len(_BANNER_SPINNER_FRAMES)]
            live.update(_render_welcome_panel(agent, frame))
        # Final settle frame: wordmark ✱. Users who eyeball the banner
        # post-startup see a stable, readable glyph.
        _time.sleep(_BANNER_FRAME_INTERVAL)
        live.update(_render_welcome_panel(agent, _BANNER_SETTLE_GLYPH))


def _print_verbose_summary(agent: Agent, console: Console) -> None:
    state = agent.state
    console.print(
        f"[dim]\\[turn {state.turn_count} \u00b7 "
        f"{state.total_tokens_used:,} tokens \u00b7 "
        f"{agent.current_model}][/dim]"
    )


def _print_post_turn_status(
    agent: Agent, console: Console, last_turn_seconds: float = 0.0,
) -> None:
    """Print a minimal turn-end checkpoint in the scrollback.

    pt's ``bottom_toolbar`` already carries the full live status
    (model / context pressure / pinned / mode / cwd) while pt owns the
    screen. The toolbar disappears during model streaming, so we still
    want a scrollback marker — but the marker must NOT duplicate the
    toolbar, or both render simultaneously when pt re-takes the screen
    and the user sees the same info twice stacked above the prompt.

    Shape: a single dim line, ``done`` + elapsed time. Nothing else.
    Everything else lives in the live toolbar.
    """
    text = Text("done", style="dim")
    if last_turn_seconds > 0:
        if last_turn_seconds < 60:
            duration = f"{last_turn_seconds:.1f}s"
        else:
            duration = f"{int(last_turn_seconds)}s"
        text.append(f"  ·  {duration}", style="dim")
    console.print(text)


async def _run_turn(
    agent: Agent, prompt: str, renderer: Renderer, console: Console,
) -> float:
    """Run one turn end-to-end; return its wall-clock duration in seconds.

    The duration is captured at the REPL layer (not on ``Agent.state``)
    because it's display-only info for the status bar — ephemeral,
    orthogonal to agent semantics. Measured with ``time.perf_counter``
    for monotonic wall time (unaffected by system clock adjustments).
    """
    from aura.cli._coordination import set_spinner_pause_callback
    from aura.cli.attachments import (
        extract_and_resolve_attachments,
        render_attachments_as_messages,
    )

    # @mention resolution happens BEFORE the spinner — the network-bound
    # MCP reads shouldn't be hidden behind "thinking…" (the user wants to
    # see that a doc was pulled before the LLM starts). If no mentions
    # match, this short-circuits to the original prompt + empty list.
    resolved_prompt, attachments = await extract_and_resolve_attachments(
        prompt, agent.mcp_manager,
    )
    attachment_messages = render_attachments_as_messages(attachments)
    # Surface each resolved attachment as a dim line so the user sees
    # exactly which resources were pulled. One line per attachment; zero
    # output when no mentions resolved (keeps the common path clean).
    for att in attachments:
        console.print(
            f"[dim]attached @{att.server}:{att.uri}"
            f" ({len(att.content)} chars)[/dim]"
        )

    spinner = ThinkingSpinner(console)
    spinner.start()
    spinner_stopped = False

    async def _stop_spinner() -> None:
        nonlocal spinner_stopped
        if not spinner_stopped:
            spinner_stopped = True
            await spinner.stop()

    # Let the permission asker (or any other mid-turn widget) pause the
    # spinner before it takes the screen. Registered for the lifetime of
    # this turn; cleared in the finally so an asker outside a turn
    # (future: proactive bg tasks?) doesn't accidentally poke a stopped
    # spinner.
    set_spinner_pause_callback(_stop_spinner)

    async def _stream() -> None:
        async for event in agent.astream(
            resolved_prompt, attachments=attachment_messages or None,
        ):
            await _stop_spinner()
            renderer.on_event(event)
        await _stop_spinner()
        renderer.finish()

    started = time.perf_counter()
    task = asyncio.create_task(_stream())
    try:
        await task
    except asyncio.CancelledError:
        await _stop_spinner()
    except KeyboardInterrupt:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await _stop_spinner()
    finally:
        await _stop_spinner()
        set_spinner_pause_callback(None)
    return time.perf_counter() - started
