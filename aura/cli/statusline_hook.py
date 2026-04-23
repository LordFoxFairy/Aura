"""Runner for the user-configured StatusLine hook.

Mirrors claude-code's ``statusLine`` hook: the user sets a shell
command in ``.aura/settings.json`` under ``permissions.statusline``; on
each toolbar paint Aura forks that command, pipes a JSON envelope on
stdin, and uses its stdout as the bar text. See
``aura/schemas/permissions.py`` for the config schema and
``aura/cli/status_bar.py`` for the envelope → HTML integration.

Stability contract — the envelope passed on stdin is v1:

    {
      "version": "1",
      "model": str | None,
      "context_window": {"size": int, "used": int, "pct": int},
      "tokens": {"input": int, "cache_read": int, "pinned_estimate": int},
      "mode": str,
      "cwd": str,
      "last_turn_seconds": float,
    }

Keys are stable lowercase snake_case. Any breaking change bumps
``version``; additive changes (new top-level fields, new sub-dict
fields) leave ``"1"`` intact so user scripts relying on the current
shape keep working.

Failure model is "fail silent, fall back": non-zero exit, timeout,
command-not-found, UnicodeDecodeError, empty output — all return
``None`` so the caller can render the default Aura toolbar. stderr is
captured but never surfaced; a noisy statusline script shouldn't vomit
escape sequences into the user's REPL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

STATUSLINE_ENVELOPE_VERSION = "1"


async def run_statusline_command(
    *,
    command: str,
    timeout_seconds: float,
    envelope: dict[str, Any],
) -> str | None:
    """Execute ``command`` through the shell, feed ``envelope`` on stdin,
    wait up to ``timeout_seconds``, and return its stdout as a string.

    Returns ``None`` on any failure (timeout, non-zero exit, empty output,
    spawn error, decoding error). The caller MUST treat ``None`` as "use
    the default render" — there is no error surface by design.

    Implementation notes:

    - ``create_subprocess_shell`` (vs ``_exec``) so the user-supplied
      string can use shell features like ``bash -c '...'``. That is the
      same contract claude-code's ``statusLine.command`` offers; forcing
      argv-splitting here would silently break shebang scripts and
      ``~`` expansion.
    - stderr is captured to ``PIPE`` purely to keep it off the user's
      TTY. We don't inspect or propagate it.
    - Timeout handling is a belt-and-braces kill + wait: ``kill()``
      sends SIGKILL, then we drain ``communicate()`` so zombie pipes
      don't leak between calls. The second await is wrapped in its
      own small timeout because a truly wedged process can still hang
      there; 1s is plenty for SIGKILL delivery.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError):
        return None

    try:
        payload = json.dumps(envelope).encode("utf-8")
    except (TypeError, ValueError):
        # Envelope contained something non-JSON-serializable. Shouldn't
        # happen for the fixed schema we build, but never crash the bar.
        proc.kill()
        await proc.wait()
        return None

    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(input=payload),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(proc.communicate(), timeout=1.0)
        return None
    except (OSError, ValueError):
        return None

    if proc.returncode != 0 or not stdout:
        return None
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Trim trailing newline only — leading whitespace may be intentional
    # (e.g. to pad the bar), and internal ANSI sequences must be preserved.
    if text.endswith("\n"):
        text = text[:-1]
    if not text:
        return None
    return text


def build_envelope(
    *,
    model: str | None,
    context_window_size: int,
    input_tokens: int,
    cache_read_tokens: int,
    pinned_estimate_tokens: int,
    mode: str,
    cwd: str,
    last_turn_seconds: float,
) -> dict[str, Any]:
    """Build the v1 JSON envelope fed to the hook on stdin.

    Centralised so both the production caller and tests assert against
    ONE schema. Kept in this module (not ``status_bar``) because the
    envelope is a property of the hook protocol, not of the rendering
    surface that happens to invoke it today.
    """
    pct = 0
    if input_tokens > 0 and context_window_size > 0:
        pct = max(1, round(input_tokens / context_window_size * 100))
    return {
        "version": STATUSLINE_ENVELOPE_VERSION,
        "model": model,
        "context_window": {
            "size": context_window_size,
            "used": input_tokens,
            "pct": pct,
        },
        "tokens": {
            "input": input_tokens,
            "cache_read": cache_read_tokens,
            "pinned_estimate": pinned_estimate_tokens,
        },
        "mode": mode,
        "cwd": cwd,
        "last_turn_seconds": last_turn_seconds,
    }
