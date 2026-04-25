"""Teammate runtime — the long-lived loop that drives one teammate Agent.

Lifetime: spawned once by :meth:`TeamManager.add_member`, exits on
``stop_event.set()`` (graceful) or ``abort.abort()`` (cascade from leader).

Loop sketch (per design §3.4):

1. If a ``seed_prompt`` was supplied at spawn, run it through the Agent
   without waiting for a mailbox message. (First-turn convenience so the
   add-member API doesn't need a separate "kick off" send.)
2. Wait for the next unseen mailbox message (off-thread polling).
3. Drain unseen messages; ack them.
4. If any of them are ``shutdown_request``, exit cleanly.
5. Concatenate text messages into one envelope-wrapped HumanMessage and
   feed it into ``Agent.astream``. Final + AssistantDelta events flow
   into the teammate's transcript.
6. Loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from aura.core.abort import AbortController, AbortException
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.teams.mailbox import Mailbox
from aura.core.teams.types import TeamMessage
from aura.schemas.events import Final

if TYPE_CHECKING:
    from aura.core.agent import Agent

#: How long to block per ``read_unseen`` poll cycle before checking the
#: stop_event again. 5s keeps shutdowns snappy without busy-spinning the
#: filesystem.
_POLL_SLICE_SEC: float = 5.0


def _format_envelope(messages: list[TeamMessage]) -> str:
    """Wrap incoming TeamMessages into a single user-facing prompt.

    ``<from-{sender}>...</from-{sender}>`` envelopes match the design
    proposal §5 wording — the model sees a clear sender attribution
    without us inventing structured tool-call shape. Multiple senders
    are concatenated in arrival order.
    """
    parts: list[str] = []
    for m in messages:
        parts.append(f"<from-{m.sender}>\n{m.body}\n</from-{m.sender}>")
    return "\n\n".join(parts)


async def _drive_one_turn(
    *,
    agent: Agent,
    prompt: str,
    abort: AbortController,
    storage: SessionStorage,
    team_id: str,
    member_name: str,
) -> str:
    """Run one ``Agent.astream`` pass; return the Final text.

    Streams every event into the teammate's transcript file via
    ``append_team_transcript``. ``Final`` is the natural-completion
    marker; ``AbortException`` short-circuits with the partial reason.
    """
    final_text = ""
    transcript = storage.team_transcript_path(team_id, member_name)
    try:
        async for event in agent.astream(prompt, abort=abort):
            # Append a one-line representation of each event for
            # post-mortem inspection. We deliberately don't try to
            # round-trip BaseMessage objects here — the teammate's own
            # SessionStorage already does that. This file is a
            # human-readable stream of "what happened in order."
            with (
                contextlib.suppress(OSError),
                transcript.open("a", encoding="utf-8") as f,
            ):
                f.write(f"{int(time.time())} {type(event).__name__} ")
                if isinstance(event, Final):
                    f.write(event.message[:500])
                f.write("\n")
            if isinstance(event, Final):
                final_text = event.message
    except AbortException:
        journal.write(
            "team_runtime_aborted",
            team_id=team_id,
            member=member_name,
        )
        raise
    return final_text


async def run_teammate(
    *,
    agent: Agent,
    team_id: str,
    member_name: str,
    storage: SessionStorage,
    stop_event: asyncio.Event,
    abort: AbortController,
    seed_prompt: str | None = None,
) -> None:
    """Long-lived loop: drain mailbox, run agent, repeat.

    Cancellation contract:

    - ``stop_event`` set externally (e.g. ``remove_member``) → loop
      exits at the next mailbox-poll boundary or after the in-flight
      turn finishes, whichever happens first.
    - ``abort.aborted`` → propagates as ``AbortException`` out of
      ``astream`` and breaks the loop with a journal entry.
    - ``asyncio.CancelledError`` → re-raised after best-effort cleanup.
    """
    mailbox = Mailbox(storage, team_id)
    journal.write(
        "team_runtime_started",
        team_id=team_id,
        member=member_name,
    )
    try:
        # Seed-prompt path: run once without waiting on the mailbox so
        # the leader's ``add_member(seed_prompt=...)`` call gets the
        # teammate moving immediately. The seed itself is NOT routed
        # through the mailbox — that would double-bill it on restart.
        if seed_prompt is not None and seed_prompt.strip():
            with contextlib.suppress(AbortException):
                await _drive_one_turn(
                    agent=agent,
                    prompt=seed_prompt,
                    abort=abort,
                    storage=storage,
                    team_id=team_id,
                    member_name=member_name,
                )
        while not stop_event.is_set() and not abort.aborted:
            # Off-thread poll — Mailbox.wait_for_new_message blocks on
            # ``time.sleep`` internally. Slicing into _POLL_SLICE_SEC
            # chunks keeps the stop_event responsive.
            try:
                has_msg = await asyncio.to_thread(
                    mailbox.wait_for_new_message,
                    member_name,
                    timeout=_POLL_SLICE_SEC,
                )
            except asyncio.CancelledError:
                raise
            if not has_msg:
                continue
            unseen = mailbox.read_unseen(member_name)
            if not unseen:
                continue
            mailbox.ack(member_name, [m.msg_id for m in unseen])
            # Look for a control message first — shutdown short-circuits.
            shutdown = next(
                (m for m in unseen if m.kind == "shutdown_request"), None,
            )
            if shutdown is not None:
                journal.write(
                    "team_runtime_shutdown",
                    team_id=team_id,
                    member=member_name,
                    sender=shutdown.sender,
                )
                # Phase A.1 — emit a real ``shutdown_response`` to the
                # leader's mailbox BEFORE breaking. Without this leg the
                # leader's ``aremove_member`` waits 5s on the timeout
                # branch and falls through to force-kill, which can
                # interrupt teammates mid-tool-call. The runtime owns
                # the ack: it has just consumed the request, knows the
                # reason, and will exit on the next statement so the
                # ack is honest. Best-effort: a mailbox write failure
                # must not crash the runtime — the leader still has
                # force-kill on timeout as the safety net.
                manager = getattr(agent, "team", None)
                if manager is not None and getattr(manager, "is_active", False):
                    with contextlib.suppress(Exception):
                        manager.send(
                            sender=member_name,
                            recipient="leader",
                            body=f"shutting down: {shutdown.body}",
                            kind="shutdown_response",
                        )
                break
            text_msgs = [m for m in unseen if m.kind == "text"]
            if not text_msgs:
                continue
            envelope = _format_envelope(text_msgs)
            try:
                await _drive_one_turn(
                    agent=agent,
                    prompt=envelope,
                    abort=abort,
                    storage=storage,
                    team_id=team_id,
                    member_name=member_name,
                )
            except AbortException:
                break
            except Exception as exc:  # noqa: BLE001
                journal.write(
                    "team_runtime_turn_failed",
                    team_id=team_id,
                    member=member_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                # Keep looping — a single turn failure shouldn't kill
                # the teammate. Phase B may add a per-member failure
                # threshold; for now the leader can /team remove.
                continue
    except asyncio.CancelledError:
        journal.write(
            "team_runtime_cancelled",
            team_id=team_id,
            member=member_name,
        )
        raise
    finally:
        journal.write(
            "team_runtime_exited",
            team_id=team_id,
            member=member_name,
        )
