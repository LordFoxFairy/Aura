"""Headless NDJSON entry point for desktop / external-frontend integrations.

Reads line-delimited JSON requests from stdin, emits one JSON event per line
to stdout. Each request is a ``{"kind": "prompt", "text": "..."}`` envelope;
each response is the JSON-serialized event from :mod:`aura.schemas.events`
plus a small ``{"event": "<name>"}`` discriminator.

Designed for the Tauri desktop frontend that spawns ``python -m
aura.desktop.headless`` as a subprocess and pipes user prompts down stdin while
streaming agent events back up stdout. Pure stdio — no prompt_toolkit,
no rich, no terminal control sequences. Stderr is reserved for fatal
errors / startup banners; routine errors come through stdout as
``{"event": "error", "message": ...}`` records.

Event shapes (all NDJSON, one per line):

- ``{"event": "ready", "session_id": "..."}`` — emitted once at startup
- ``{"event": "assistant_delta", "text": "..."}`` — streaming model text
- ``{"event": "tool_call_started", "name": "...", "args": {...}, "id": "..."}``
- ``{"event": "tool_call_progress", "id": "...", "chunk": {...}}``
- ``{"event": "tool_call_completed", "id": "...", "ok": bool, "result": {...}}``
- ``{"event": "final", "message": "...", "reason": "..."}`` — turn ended
- ``{"event": "error", "message": "..."}`` — fatal turn error
- ``{"event": "exited"}`` — emitted right before the process closes stdin

The CLI is single-tenant (one Agent per process, one prompt at a time).
For multi-session use, the desktop spawns multiple subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool

from aura.config.loader import load_config
from aura.core.agent import Agent
from aura.core.hooks import HookChain
from aura.core.hooks.permission import AskerResponse, make_permission_hook
from aura.core.llm import make_model_for_spec
from aura.core.permissions import SafetyPolicy
from aura.core.permissions import store as perm_store
from aura.core.permissions.defaults import DEFAULT_ALLOW_RULES
from aura.core.permissions.rule import Rule
from aura.core.permissions.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
)
from aura.core.permissions.session import RuleSet, SessionRuleSet
from aura.core.persistence.storage import SessionStorage
from aura.schemas.events import (
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)


def _emit(payload: dict[str, Any]) -> None:
    """Write one NDJSON line to stdout + flush.

    Tauri reads stdout line-by-line; an unflushed line stays in Python's
    pipe buffer, leaving the frontend "stuck" mid-turn. Flush after every
    write so streaming feels real-time.
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class IpcAsker:
    """``PermissionAsker`` impl that round-trips through stdio NDJSON.

    Phase 1.5 — desktop-side permission UI. When a tool needs user
    consent, the asker:

    1. mints a request id (so multiple concurrent prompts are unambiguous)
    2. emits a ``permission_request`` event on stdout
    3. parks on a ``Future`` keyed by that id
    4. resumes when ``feed_response(...)`` is called from the stdin
       dispatcher with a matching ``permission_response`` payload

    The CLI's stdin-based asker (``aura.cli.permission``) is untouched —
    this is a parallel implementation for the headless / Tauri path.
    A single Aura process picks one asker at construction time; the
    desktop bridge wires this one, the CLI wires the prompt_toolkit one.

    Mapping to :class:`AskerResponse`:

    - ``choice="accept"``  → one-shot allow (session-scoped, no rule installed)
    - ``choice="always"``  → install ``rule_hint`` for the rest of the
      session so subsequent matching calls auto-allow
    - ``choice="deny"``    → reject this call; model sees a tool error

    The frontend modal is responsible for surfacing the rule-hint string
    to the user so they know exactly what "always" will install.
    """

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def __call__(
        self,
        *,
        tool: BaseTool,
        args: dict[str, Any],
        rule_hint: Rule,
    ) -> AskerResponse:
        req_id = uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut

        # Render args so the frontend can show a preview. JSON-friendly:
        # convert non-serializable objects to their str form rather than
        # exploding the whole event.
        try:
            safe_args = json.loads(json.dumps(args, default=str))
        except (TypeError, ValueError):
            safe_args = {"_repr": repr(args)}

        _emit({
            "event": "permission_request",
            "id": req_id,
            "tool": tool.name,
            "args": safe_args,
            "rule_hint": rule_hint.to_string(),
            # ``is_destructive`` lets the modal pick a louder visual treatment
            # for destructive tools (red outline vs yellow). Falls back to
            # True (conservative) when the tool didn't declare its capability.
            "is_destructive": bool(
                (tool.metadata or {}).get("is_destructive", True),
            ),
        })

        try:
            response = await fut
        finally:
            self._pending.pop(req_id, None)

        choice = response.get("choice")
        feedback = str(response.get("feedback") or "")
        if choice == "always":
            return AskerResponse(
                choice="always",
                scope="session",
                rule=rule_hint,
                feedback=feedback,
            )
        if choice == "accept":
            return AskerResponse(
                choice="accept", scope="session", feedback=feedback,
            )
        # Anything else (deny, missing, malformed) → deny. Defensive default.
        return AskerResponse(
            choice="deny", scope="session", feedback=feedback,
        )

    def feed_response(self, payload: dict[str, Any]) -> bool:
        """Resolve the Future for ``payload['id']``. Returns False on miss.

        Frontend can race a permission_response after the request already
        timed out / was cancelled; missing-id is silent (not an error)
        because the request is gone and the frontend has nothing useful
        to do with the late ack.
        """
        req_id = payload.get("id")
        if not isinstance(req_id, str):
            return False
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return False
        fut.set_result(payload)
        return True


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Map an :mod:`aura.schemas.events` dataclass to its NDJSON shape.

    Field names mirror the dataclass attributes verbatim so the frontend
    type definitions stay aligned with the Python source of truth.
    """
    if isinstance(event, AssistantDelta):
        return {"event": "assistant_delta", "text": event.text}
    if isinstance(event, ToolCallStarted):
        return {
            "event": "tool_call_started",
            "name": event.name,
            "input": event.input,
        }
    if isinstance(event, ToolCallProgress):
        return {
            "event": "tool_call_progress",
            "name": event.name,
            "stream": event.stream,
            "chunk": event.chunk,
        }
    if isinstance(event, ToolCallCompleted):
        return {
            "event": "tool_call_completed",
            "name": event.name,
            "output": event.output,
            "error": event.error,
        }
    if isinstance(event, Final):
        return {
            "event": "final",
            "message": event.message,
            "reason": getattr(event, "reason", "natural"),
        }
    # Unknown event types — surface the class name so the frontend can
    # fall back to a generic "info" line instead of swallowing.
    return {"event": "unknown", "type": type(event).__name__}


async def _run() -> int:
    cfg = load_config()
    spec = cfg.router.get("default", "")
    if not spec:
        _emit({
            "event": "error",
            "message": "config.router['default'] is missing — cannot start headless",
        })
        return 1
    model = make_model_for_spec(spec, cfg)
    storage_path = Path(cfg.storage.path).expanduser()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = SessionStorage(storage_path)
    # Permission setup — mirrors the CLI's plumbing (see
    # aura/cli/__main__.py) but with our IpcAsker instead of the
    # prompt_toolkit asker.
    project_root = Path.cwd()
    try:
        perm_cfg = perm_store.load(project_root)
        known_tools = list(cfg.tools.enabled) + ["mcp__*"]
        disk_rules = perm_store.load_ruleset(
            project_root, known_tool_names=known_tools,
        )
    except Exception as exc:  # noqa: BLE001
        _emit({
            "event": "error",
            "message": f"permissions config: {type(exc).__name__}: {exc}",
        })
        return 1
    ruleset = RuleSet(rules=disk_rules.rules + DEFAULT_ALLOW_RULES)
    safety_policy = SafetyPolicy(
        protected_writes=DEFAULT_PROTECTED_WRITES,
        protected_reads=DEFAULT_PROTECTED_READS,
        exempt=tuple(perm_cfg.safety_exempt),
    )
    session = SessionRuleSet()
    asker = IpcAsker()
    from typing import Literal
    Mode = Literal["default", "bypass", "plan", "accept_edits"]
    mode: Mode = perm_cfg.mode

    def _live_mode() -> Mode:
        # The desktop has no /mode slash command yet; mode stays whatever
        # the config said at startup. Phase 2 will surface a mode toggle
        # in the status bar and route changes back through here.
        return mode

    permission_hook = make_permission_hook(
        asker=asker,
        session=session,
        rules=ruleset,
        project_root=project_root,
        mode=_live_mode,
        safety=safety_policy,
    )
    hooks = HookChain(pre_tool=[permission_hook])

    agent = Agent(
        config=cfg,
        model=model,
        storage=storage,
        hooks=hooks,
        session_rules=session,
        mode=mode,
    )
    _emit({"event": "ready", "session_id": agent.session_id, "model": spec})

    # Track the in-flight astream task so a ``permission_response`` from
    # stdin can wake up the asker even while a turn is mid-flight.
    turn_task: asyncio.Task[None] | None = None

    async def _drive_turn(text: str) -> None:
        try:
            async for event in agent.astream(text):
                _emit(_event_to_dict(event))
        except Exception as exc:  # noqa: BLE001
            _emit({
                "event": "error",
                "message": f"{type(exc).__name__}: {exc}",
            })

    try:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            line = await reader.readline()
            if not line:
                # stdin closed — the frontend exited.
                break
            try:
                request = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError as exc:
                _emit({"event": "error", "message": f"bad request: {exc}"})
                continue

            kind = request.get("kind")
            if kind == "permission_response":
                # Resolve the asker Future immediately — this MUST run
                # while a turn is in flight (the asker is awaiting the
                # response from inside agent.astream). Don't queue it
                # behind the turn_task.
                if not asker.feed_response(request):
                    _emit({
                        "event": "error",
                        "message": (
                            f"no pending permission request for id="
                            f"{request.get('id')!r}"
                        ),
                    })
                continue

            if kind == "prompt":
                text = request.get("text", "")
                if not isinstance(text, str) or not text:
                    _emit({"event": "error", "message": "empty prompt"})
                    continue
                # Park the turn on a task so we can keep reading stdin
                # for permission_response while it runs. The task is
                # awaited at the bottom of the loop iteration so we
                # don't process two prompts concurrently (single-tenant).
                turn_task = asyncio.create_task(_drive_turn(text))
                # Drain any permission_responses that arrive WHILE the
                # turn is running. We poll readline with a short timeout
                # so we can also notice turn_task completion.
                while not turn_task.done():
                    try:
                        line = await asyncio.wait_for(
                            reader.readline(), timeout=0.1,
                        )
                    except TimeoutError:
                        continue
                    if not line:
                        # stdin closed mid-turn — let the turn finish
                        # but stop reading.
                        break
                    try:
                        sub = json.loads(line.decode("utf-8").strip())
                    except json.JSONDecodeError as exc:
                        _emit({
                            "event": "error",
                            "message": f"bad request mid-turn: {exc}",
                        })
                        continue
                    if sub.get("kind") == "permission_response":
                        asker.feed_response(sub)
                    else:
                        _emit({
                            "event": "error",
                            "message": (
                                "only permission_response accepted "
                                f"mid-turn; got kind={sub.get('kind')!r}"
                            ),
                        })
                await turn_task
                turn_task = None
                continue

            _emit({
                "event": "error",
                "message": f"unsupported request kind: {kind!r}",
            })
    finally:
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
        await agent.aclose()
        _emit({"event": "exited"})
    return 0


def main() -> int:
    """Entry point — bootstrap an Agent and pipe stdin → astream → stdout."""
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        _emit({"event": "exited"})
        return 130


if __name__ == "__main__":
    sys.exit(main())
