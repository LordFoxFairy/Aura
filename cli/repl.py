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
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from aura import __version__
from aura.application.commands import CommandRegistry
from aura.application.commands.registry import build_default_registry, dispatch
from aura.core.agent import Agent
from aura.infrastructure.persistence import journal
from cli.completion import SlashCommandCompleter, resolve_history_path
from cli.render import Renderer

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
    exit. One instance is allocated per REPL session (or per test
    binding) and shared with the c-c keybinding via closure.
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
    console: Console | None,
    ctrl_c_state: _CtrlCState | None = None,
) -> KeyBindings:
    """Build a KeyBindings carrying Aura's prompt-level bindings.

    Bindings:

    - ``s-tab`` → cycle permission mode (default → accept_edits → plan);
      under bypass it's a silent no-op (bypass is sticky).
    - ``escape`` → reset mode to default. **Non-eager** so the ESC prefix
      of meta-key sequences (Alt/Meta+Enter arrives as ESC then CR) is
      still available for the bindings below.
    - ``escape, enter`` and ``c-j`` → insert literal ``\\n``. Together
      these cover Alt/Meta+Enter on macOS Terminal.app + iTerm2 plus the
      Shift+Enter→Ctrl+J remap many Linux terminals expose.
    - ``c-c`` → three-state handler (claude-code parity with
      ``useExitOnCtrlCD.ts`` + ``useDoublePress.ts``):
      buffer non-empty → clear; bare first press → arm the
      double-press window; bare second press within
      :data:`_CTRL_C_DOUBLE_PRESS_SECONDS` → raise ``KeyboardInterrupt``.

    Plain Enter keeps its pt default "accept-line"; we deliberately do
    NOT set ``multiline=True`` on the session. ``console`` is reserved
    in the signature for future bindings that need out-of-band output.
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
            state.last_press_at = 0.0
            event.app.exit(exception=KeyboardInterrupt())
            return
        # Case 2: first bare Ctrl+C — arm the double-press window.
        state.last_press_at = now
        event.app.invalidate()

    return kb


def _build_prompt_session(
    registry: CommandRegistry,
    agent: Agent | None = None,
    console: Console | None = None,
) -> PromptSession[str]:
    """Construct a PromptSession wired with history and slash-completion.

    - ``FileHistory`` at ``~/.aura/history`` → up-arrow cycles across sessions.
    - ``search_ignore_case=True`` → Ctrl+R reverse search, case-insensitive.
    - ``SlashCommandCompleter`` with a live registry getter → Skill / MCP
      commands registered after PromptSession construction still complete.
    - ``complete_while_typing=True`` → menu pops the moment the user types
      ``/`` (the completer filters by leading slash so prose never triggers).

    ``agent=None`` skips installing Aura-specific keybindings (mode cycle /
    Ctrl+C double-press) — keeps the function usable in tests that only
    exercise history + completion wiring.
    """
    history = FileHistory(str(resolve_history_path()))
    completer = SlashCommandCompleter(lambda: registry)
    key_bindings = (
        _build_mode_key_bindings(agent, console)
        if agent is not None and console is not None
        else None
    )
    return PromptSession(
        history=history,
        completer=completer,
        complete_while_typing=True,
        search_ignore_case=True,
        key_bindings=key_bindings,
    )


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
) -> None:
    journal.write("repl_started")
    _console = console if console is not None else Console()
    renderer = Renderer(_console)
    registry = build_default_registry(agent=agent)

    # Wall-clock duration of the most recent turn. Kept in a single-element
    # list so a future status-line surface can sample it without changing
    # the closure shape; today only the post-turn "done · 1.2s" line reads it.
    last_turn_seconds: list[float] = [0.0]

    # Resolution order for the input function:
    # 1. Explicit ``input_fn`` override (tests / non-interactive callers).
    # 2. If stdin is a TTY, build a PromptSession (history, completion, Ctrl+R).
    # 3. Otherwise fall back to plain ``input()`` so piped / dumb terminals
    #    don't hang waiting on a prompt_toolkit renderer they can't drive.
    if input_fn is not None:
        _input: InputFn = input_fn
    elif sys.stdin.isatty():
        _input = _make_prompt_session_input(
            _build_prompt_session(
                registry,
                agent=agent,
                console=_console,
            )
        )
    else:
        _input = _default_input

    _print_welcome(agent, _console)

    # In bypass mode the startup banner scrolls off after a few turns;
    # encode bypass into the prompt string so every line reminds the user
    # they're in "allow-everything" mode.
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
            if result.kind == "view":
                # Modal-style display: wrap text in a framed panel and
                # block until the user hits Enter.
                _render_view(_console, result.text)
                continue
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
            # user's interactive session.
            journal.write(
                "turn_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            _console.print(
                f"[red]turn failed: {type(exc).__name__}: {exc}[/red]"
            )

        # Post-turn status checkpoint — a single dim "done · 1.2s" line.
        _print_post_turn_status(agent, _console, last_turn_seconds[0])

        # V14 — when the user has /team-entered a team, print a thin
        # status line between prompts so the active context stays visible.
        _print_active_team_status(agent, _console)

        if verbose:
            _print_verbose_summary(agent, _console)


#: Frame set for the leading glyph on the welcome banner. Palindromic so the
#: animation bounces rather than resets. Matches claude-code's darwin spinner
#: glyph family (``src/components/Spinner/utils.ts::getDefaultCharacters``)
#: for visual continuity between startup banner and any future in-turn
#: animation that wants the same vocabulary.
_BANNER_SPINNER_FRAMES: tuple[str, ...] = (
    "·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢",
)
#: Total on-screen time for the welcome animation.
_BANNER_ANIMATION_SECONDS: float = 1.2
_BANNER_FRAME_INTERVAL: float = 0.12
#: The glyph the banner SETTLES on after animation — the ``✱`` matches the
#: wordmark we've shipped since v0.1 and renders cleanly in every terminal.
_BANNER_SETTLE_GLYPH: str = "✱"


def _render_view(console: Console, text: str) -> None:
    """Render a ``kind="view"`` command output as a modal-style panel.

    Wraps ``text`` in a dim-bordered :class:`rich.panel.Panel`, then blocks
    on ``console.input`` until the user hits Enter (or Ctrl+C / Ctrl+D —
    both dismiss silently). Empty ``text`` is valid: commands that already
    emitted their own output still want the "press Enter" pause.
    """
    stripped = text.strip("\n")
    if stripped:
        console.print(Panel(stripped, border_style="dim", padding=(0, 1)))
    try:
        console.input("[dim](press Enter to continue) [/dim]")
    except (EOFError, KeyboardInterrupt, OSError):
        # Silent dismiss — Ctrl+C / Ctrl+D should not crash the REPL.
        # ``OSError``: pytest capture wraps stdin with a reader that
        # raises on read during non-interactive runs.
        console.print()


def _render_welcome_panel(agent: Agent, glyph: str) -> Panel:
    """Build the welcome Panel with the given leading glyph."""
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

    The leading glyph animates through a spinner-style frame set for a
    brief window after launch, then settles on ``✱``. Non-TTY callers
    (``console.is_terminal == False``) short-circuit the animation and
    print the settled banner directly so StringIO tests still pass.
    """
    if not console.is_terminal:
        console.print(_render_welcome_panel(agent, _BANNER_SETTLE_GLYPH))
        return

    import time as _time

    from rich.live import Live

    total_frames = max(
        1, int(_BANNER_ANIMATION_SECONDS / _BANNER_FRAME_INTERVAL),
    )
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
        # Final settle frame: wordmark ✱.
        _time.sleep(_BANNER_FRAME_INTERVAL)
        live.update(_render_welcome_panel(agent, _BANNER_SETTLE_GLYPH))


def _print_verbose_summary(agent: Agent, console: Console) -> None:
    state = agent.state
    console.print(
        f"[dim]\\[turn {state.turn_count} · "
        f"{state.total_tokens_used:,} tokens · "
        f"{agent.current_model}][/dim]"
    )


def _print_post_turn_status(
    agent: Agent, console: Console, last_turn_seconds: float = 0.0,
) -> None:
    """Print a minimal turn-end checkpoint in the scrollback.

    Shape: a single dim line, ``done`` + elapsed time. Nothing else.
    """
    del agent  # reserved — future per-agent decorations
    text = Text("done", style="dim")
    if last_turn_seconds > 0:
        if last_turn_seconds < 60:
            duration = f"{last_turn_seconds:.1f}s"
        else:
            duration = f"{int(last_turn_seconds)}s"
        text.append(f"  ·  {duration}", style="dim")
    console.print(text)


def _print_active_team_status(agent: Agent, console: Console) -> None:
    """Print a one-line ``· in team: <name> ·`` reminder when active."""
    active_id = agent.state.slots.active_team
    if not active_id:
        return
    label = active_id
    mgr = getattr(agent, "_team_manager", None)
    if mgr is not None:
        live = getattr(mgr, "team", None)
        if live is not None and getattr(live, "team_id", None) == active_id:
            label = live.name
    console.print(Text(f"· in team: {label} ·", style="dim"))


async def _run_turn(
    agent: Agent, prompt: str, renderer: Renderer, console: Console,
) -> float:
    """Run one turn end-to-end; return its wall-clock duration in seconds.

    The renderer's normal event stream is the sole feedback surface —
    no spinner, no attachments preprocessing, no bottom toolbar.
    Cancel / KeyboardInterrupt mid-stream cleanly cancels the underlying
    astream task without surfacing a traceback to the REPL loop.
    """
    del console  # reserved — future per-turn console hooks

    async def _stream() -> None:
        async for event in agent.astream(prompt):
            if isinstance(event, dict):
                # Phase 4 Task 4 — wire-format compact events flow
                # alongside typed AgentEvent. The CLI renderer is
                # dataclass-only; compact lifecycle is silent here.
                continue
            renderer.on_event(event)
        renderer.finish()

    started = time.perf_counter()
    task = asyncio.create_task(_stream())
    try:
        await task
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return time.perf_counter() - started
