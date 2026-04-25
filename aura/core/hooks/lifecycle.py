"""Lifecycle hook glue — settings.json command adapters.

F-04-014. The 4 lifecycle event protocols themselves
(:class:`SessionStartHook` / :class:`UserPromptSubmitHook` /
:class:`NotificationHook` / :class:`StopHook`) live in
``aura.core.hooks.__init__`` next to the existing hook protocols so the
import surface is uniform. This module owns the glue that lets users
register external hook commands declaratively via ``settings.json``::

    {
      "hooks": {
        "session_start":      [{"command": "/usr/local/bin/aura-on-start"}],
        "user_prompt_submit": [{"command": "python /tmp/pii_strip.py"}],
        "notification":       [{"command": "slack-notify"}],
        "stop":               [{"command": "aura-flush-metrics"}]
      }
    }

For each ``{command, timeout_ms?}`` entry we build a small async adapter
that:

1. Spawns the command via :func:`asyncio.create_subprocess_shell` —
   same shape as the existing ``statusline_hook.py`` so users don't have
   to learn a second envelope shape.
2. Pipes a JSON envelope on stdin (the hook's payload, schema-versioned).
3. Reads stdout up to the timeout. Non-zero exit / timeout / decode
   error == "no result"; for ``user_prompt_submit`` a non-empty stdout
   becomes the rewritten prompt, for the other 3 events stdout is
   ignored (the events are pure side-effect).
4. Failures are journaled + swallowed — a broken external hook MUST
   NOT take down the agent.

The wiring (calling :func:`build_lifecycle_hooks_from_config` and adding
the resulting hooks to the agent's :class:`HookChain`) is done at CLI
startup; SDK consumers can register :class:`SessionStartHook` etc.
directly in code without going through this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from aura.core.hooks import (
    HookChain,
    NotificationHook,
    NotificationKind,
    SessionStartHook,
    StopHook,
    UserPromptSubmitHook,
    UserPromptSubmitOutcome,
)
from aura.core.persistence import journal

# Envelope schema version for stdin payloads. Bump when the field set
# changes incompatibly so external command authors can target a known
# shape. Currently v1 == "all field names from the protocol payload, plus
# an ``event`` discriminator".
LIFECYCLE_ENVELOPE_VERSION = "1"

# Default subprocess timeout — generous enough for a network notify call
# (slack webhook ~ 1-2s) but bounded so the lifecycle event isn't held
# hostage by a wedged hook. Per-hook override via ``timeout_ms`` in the
# settings entry.
_DEFAULT_TIMEOUT_MS = 5_000


async def _run_command(
    *,
    command: str,
    timeout_ms: int,
    envelope: dict[str, Any],
) -> str | None:
    """Run ``command`` with ``envelope`` JSON on stdin; return stdout or None.

    Mirrors :func:`aura.cli.statusline_hook.run_statusline_command`'s
    contract verbatim — ``None`` on any failure (timeout, non-zero exit,
    spawn error, decode error). Caller decides what to do with stdout
    (use it as the rewritten prompt for ``user_prompt_submit``; ignore
    it for the other events).
    """
    timeout_seconds = max(0.05, timeout_ms / 1000.0)
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

    if proc.returncode != 0:
        return None
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if text.endswith("\n"):
        text = text[:-1]
    return text or None


def _make_session_start_command_hook(
    *, command: str, timeout_ms: int,
) -> SessionStartHook:
    async def _hook(
        *,
        session_id: str,
        mode: str,
        cwd: Any,
        model_name: str,
        state: Any,
        **_: Any,
    ) -> None:
        envelope = {
            "envelope_version": LIFECYCLE_ENVELOPE_VERSION,
            "event": "session_start",
            "session_id": session_id,
            "mode": mode,
            "cwd": str(cwd),
            "model_name": model_name,
        }
        await _run_command(
            command=command, timeout_ms=timeout_ms, envelope=envelope,
        )

    return _hook


def _make_user_prompt_submit_command_hook(
    *, command: str, timeout_ms: int,
) -> UserPromptSubmitHook:
    async def _hook(
        *,
        session_id: str,
        turn_count: int,
        user_text: str,
        state: Any,
        **_: Any,
    ) -> UserPromptSubmitOutcome:
        envelope = {
            "envelope_version": LIFECYCLE_ENVELOPE_VERSION,
            "event": "user_prompt_submit",
            "session_id": session_id,
            "turn_count": turn_count,
            "user_text": user_text,
        }
        rewritten = await _run_command(
            command=command, timeout_ms=timeout_ms, envelope=envelope,
        )
        # Empty / failed → passthrough (None). Non-empty stdout → rewrite.
        if not rewritten:
            return UserPromptSubmitOutcome(prompt=None)
        return UserPromptSubmitOutcome(prompt=rewritten)

    return _hook


def _make_notification_command_hook(
    *, command: str, timeout_ms: int,
) -> NotificationHook:
    async def _hook(
        *,
        session_id: str,
        kind: NotificationKind,
        body: str,
        state: Any,
        **_: Any,
    ) -> None:
        envelope = {
            "envelope_version": LIFECYCLE_ENVELOPE_VERSION,
            "event": "notification",
            "session_id": session_id,
            "kind": kind,
            "body": body,
        }
        await _run_command(
            command=command, timeout_ms=timeout_ms, envelope=envelope,
        )

    return _hook


def _make_stop_command_hook(
    *, command: str, timeout_ms: int,
) -> StopHook:
    async def _hook(
        *,
        session_id: str,
        reason: str,
        turn_count: int,
        state: Any,
        **_: Any,
    ) -> None:
        envelope = {
            "envelope_version": LIFECYCLE_ENVELOPE_VERSION,
            "event": "stop",
            "session_id": session_id,
            "reason": reason,
            "turn_count": turn_count,
        }
        await _run_command(
            command=command, timeout_ms=timeout_ms, envelope=envelope,
        )

    return _hook


def build_lifecycle_hooks_from_config(
    raw: dict[str, Any] | None,
) -> HookChain:
    """Translate ``settings.json``'s ``hooks`` section into a HookChain.

    ``raw`` is the value of ``permissions.hooks`` (or equivalent) from
    settings.json — a dict mapping event name to a list of
    ``{command: str, timeout_ms?: int}`` entries. Returns a fresh
    HookChain with the corresponding subprocess adapters in the matching
    slots; callers ``.merge()`` it onto the agent's chain.

    Unknown event keys are journaled + skipped (forward-compat: a future
    Aura release can introduce a new event without crashing older
    settings files; an older Aura release seeing a future event just
    ignores it).

    Malformed entries (missing ``command``, non-string values, etc.)
    journal + skip per-entry. Same lenient stance the deny / ask
    rule loaders take — the operator's environment may be partially
    misconfigured, but we shouldn't refuse to start.
    """
    chain = HookChain()
    if not raw:
        return chain

    builders: dict[str, Any] = {
        "session_start": (
            chain.session_start,
            _make_session_start_command_hook,
        ),
        "user_prompt_submit": (
            chain.user_prompt_submit,
            _make_user_prompt_submit_command_hook,
        ),
        "notification": (
            chain.notification,
            _make_notification_command_hook,
        ),
        "stop": (chain.stop, _make_stop_command_hook),
    }

    for event_name, entries in raw.items():
        if event_name not in builders:
            journal.write(
                "lifecycle_hook_config_skipped",
                event=event_name,
                reason="unknown_event",
            )
            continue
        if not isinstance(entries, list):
            journal.write(
                "lifecycle_hook_config_skipped",
                event=event_name,
                reason="not_a_list",
            )
            continue
        slot, factory = builders[event_name]
        for entry in entries:
            if not isinstance(entry, dict):
                journal.write(
                    "lifecycle_hook_config_skipped",
                    event=event_name,
                    reason="entry_not_a_dict",
                )
                continue
            command = entry.get("command")
            if not isinstance(command, str) or not command.strip():
                journal.write(
                    "lifecycle_hook_config_skipped",
                    event=event_name,
                    reason="missing_or_empty_command",
                )
                continue
            timeout_ms_raw = entry.get("timeout_ms", _DEFAULT_TIMEOUT_MS)
            try:
                timeout_ms = int(timeout_ms_raw)
            except (TypeError, ValueError):
                timeout_ms = _DEFAULT_TIMEOUT_MS
            slot.append(
                factory(command=command, timeout_ms=timeout_ms),
            )

    return chain


__all__ = [
    "LIFECYCLE_ENVELOPE_VERSION",
    "build_lifecycle_hooks_from_config",
]
