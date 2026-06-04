"""Async REPL — the main loop that drives the agent from user input."""

from __future__ import annotations

import asyncio
import contextlib
import random
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from aura import __version__
from aura.application.commands import CommandRegistry
from aura.application.commands.factory import build_default_registry
from aura.application.commands.registry import dispatch
from aura.application.session import AgentSession
from aura.infrastructure.persistence import journal
from cli.completion import SlashCommandCompleter, resolve_history_path
from cli.render import Renderer

InputFn = Callable[[str], Awaitable[str]]


#: Shift+tab cycle — ``bypass`` excluded; only --bypass-permissions can enable it.
_MODE_CYCLE: tuple[str, ...] = ("default", "accept_edits", "plan")


#: Window for the second Ctrl+C to confirm exit — tuned to reject accidental double-taps.
_CTRL_C_DOUBLE_PRESS_SECONDS: float = 0.8


class _CtrlCState:
    __slots__ = ("last_press_at",)

    def __init__(self) -> None:
        # 0.0 sentinel makes the first press always look "stale" to hint_active.
        self.last_press_at: float = 0.0

    def hint_active(self, now: float) -> bool:
        return (
            self.last_press_at > 0.0
            and (now - self.last_press_at) <= _CTRL_C_DOUBLE_PRESS_SECONDS
        )


#: Tips rotated on the welcome banner — must only reference shipped features.
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
    # Modes outside the cycle (e.g. ``bypass``) pass through unchanged.
    if current not in _MODE_CYCLE:
        return current
    idx = _MODE_CYCLE.index(current)
    return _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]


def _build_mode_key_bindings(
    agent: AgentSession,
    console: Console | None,
    ctrl_c_state: _CtrlCState | None = None,
) -> KeyBindings:
    kb = KeyBindings()
    del console  # reserved for future bindings.
    state = ctrl_c_state if ctrl_c_state is not None else _CtrlCState()

    @kb.add("s-tab")
    def _(event: KeyPressEvent) -> None:
        current = agent.mode
        if current == "bypass":
            return
        agent.set_mode(_cycle_mode(current))
        event.app.invalidate()

    @kb.add("escape")
    def _(event: KeyPressEvent) -> None:
        # Non-eager so meta-key sequences (escape, enter) still match.
        if agent.mode == "bypass" or agent.mode == "default":
            return
        agent.set_mode("default")
        event.app.invalidate()

    @kb.add("escape", "enter")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-j")
    def _(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-c")
    def _(event: KeyPressEvent) -> None:
        # Three-state: text → clear; bare first → arm; bare second within window → exit.
        buffer = event.current_buffer
        now = time.monotonic()
        if buffer.text:
            buffer.reset()
            event.app.invalidate()
            return
        if state.hint_active(now):
            state.last_press_at = 0.0
            event.app.exit(exception=KeyboardInterrupt())
            return
        state.last_press_at = now
        event.app.invalidate()

    return kb


def _build_prompt_session(
    registry: CommandRegistry,
    agent: AgentSession | None = None,
    console: Console | None = None,
) -> PromptSession[str]:
    # ``agent=None`` skips Aura-specific bindings so history+completion tests can reuse this.
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
    agent: AgentSession,
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

    # Single-element list so a future status-line can sample without changing closure shape.
    last_turn_seconds: list[float] = [0.0]

    # Non-TTY paths fall back to plain input() — pt renderer can't drive a dumb terminal.
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

    # Encode bypass into the prompt so every line reminds the user it's allow-everything.
    prompt_str = "\x1b[31maura[!bypass]>\x1b[0m " if bypass else "aura> "

    while True:
        try:
            line = await _input(prompt_str)
        except (EOFError, KeyboardInterrupt):
            journal.write("repl_exit", reason="eof_or_ctrlc")
            _console.print()
            return

        # Providers 400 on empty HumanMessage — re-prompt instead of round-tripping.
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
            match result.kind:
                case "exit":
                    journal.write("repl_exit", reason="slash_exit")
                    return
                case "view":
                    _render_view(_console, result.text)
                    continue
                case _:
                    if result.text:
                        _console.print(result.text)
                    continue

        try:
            last_turn_seconds[0] = await _run_turn(
                agent, line, renderer, _console,
            )
        except Exception as exc:  # noqa: BLE001 — REPL resilience; BaseException still propagates.
            journal.write(
                "turn_failed",
                detail=f"{type(exc).__name__}: {exc}",
            )
            _console.print(
                f"[red]turn failed: {type(exc).__name__}: {exc}[/red]"
            )

        _print_post_turn_status(agent, _console, last_turn_seconds[0])
        _print_active_team_status(agent, _console)

        if verbose:
            _print_verbose_summary(agent, _console)


#: Palindromic so the welcome animation bounces rather than resets.
_BANNER_SPINNER_FRAMES: tuple[str, ...] = (
    "·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢",
)
_BANNER_ANIMATION_SECONDS: float = 1.2
_BANNER_FRAME_INTERVAL: float = 0.12
_BANNER_SETTLE_GLYPH: str = "✱"


def _render_view(console: Console, text: str) -> None:
    stripped = text.strip("\n")
    if stripped:
        console.print(Panel(stripped, border_style="dim", padding=(0, 1)))
    try:
        console.input("[dim](press Enter to continue) [/dim]")
    except (EOFError, KeyboardInterrupt, OSError):
        # OSError covers pytest's capturing stdin that raises on non-interactive runs.
        console.print()


def _render_welcome_panel(agent: AgentSession, glyph: str) -> Panel:
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


def _print_welcome(agent: AgentSession, console: Console) -> None:
    # Non-TTY callers short-circuit the spinner animation.
    if not console.is_terminal:
        console.print(_render_welcome_panel(agent, _BANNER_SETTLE_GLYPH))
        return

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
            time.sleep(_BANNER_FRAME_INTERVAL)
            frame = _BANNER_SPINNER_FRAMES[i % len(_BANNER_SPINNER_FRAMES)]
            live.update(_render_welcome_panel(agent, frame))
        time.sleep(_BANNER_FRAME_INTERVAL)
        live.update(_render_welcome_panel(agent, _BANNER_SETTLE_GLYPH))


def _print_verbose_summary(agent: AgentSession, console: Console) -> None:
    state = agent.state
    console.print(
        f"[dim]\\[turn {state.turn_count} · "
        f"{state.total_tokens_used:,} tokens · "
        f"{agent.current_model}][/dim]"
    )


def _print_post_turn_status(
    agent: AgentSession, console: Console, last_turn_seconds: float = 0.0,
) -> None:
    del agent  # reserved for future per-agent decorations
    text = Text("done", style="dim")
    if last_turn_seconds > 0:
        if last_turn_seconds < 60:
            duration = f"{last_turn_seconds:.1f}s"
        else:
            duration = f"{int(last_turn_seconds)}s"
        text.append(f"  ·  {duration}", style="dim")
    console.print(text)


def _print_active_team_status(agent: AgentSession, console: Console) -> None:
    active_id = agent.state.slots.active_team
    if not active_id:
        return
    label = active_id
    port = agent.team
    live = port.team if port is not None else None
    if live is not None and live.team_id == active_id:
        label = live.name
    console.print(Text(f"· in team: {label} ·", style="dim"))


async def _run_turn(
    agent: AgentSession, prompt: str, renderer: Renderer, console: Console,
) -> float:
    del console  # reserved for future per-turn console hooks

    async def _stream() -> None:
        async for event in agent.astream(prompt):
            # Wire-format compact dicts are silent for the CLI renderer (dataclass-only).
            if isinstance(event, dict):
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
