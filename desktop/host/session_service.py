"""Shared session driver for external desktop/front-end wrappers.

This module owns the transport-neutral Python session behavior for the current
headless desktop path: Agent construction, permission rendezvous, prompt
submission, and canonical wire-event emission. Front-end specific wrappers can
provide their own request readers and event emitters without re-owning the
session logic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from langchain_core.tools import BaseTool

from aura.application.hooks import HookChain
from aura.application.hooks.permission import make_permission_hook
from aura.application.permission.asker import AskerResponse
from aura.config.loader import load_config
from aura.core.agent import Agent
from aura.domain.permission.defaults import DEFAULT_ALLOW_RULES
from aura.domain.permission.rule import Rule
from aura.domain.permission.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
    SafetyPolicy,
)
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.infrastructure import permission_store as perm_store
from aura.infrastructure.llm import make_model_for_spec
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.stream import stream_agent_wire
from aura.infrastructure.wire.wire import agent_state_to_wire, permission_request_to_wire
from aura.schemas.tool_meta_access import meta_dict


class EventEmitter(Protocol):
    def __call__(self, payload: dict[str, Any]) -> None: ...


class RequestReader(Protocol):
    async def readline(self) -> bytes: ...


class IpcAsker:
    """Permission asker that round-trips through external request/response I/O."""

    def __init__(self, *, emit: EventEmitter) -> None:
        self._emit = emit
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

        self._emit(dict(permission_request_to_wire(
            request_id=req_id,
            tool=tool.name,
            args=args,
            rule_hint=rule_hint.to_string(),
            is_destructive=bool(meta_dict(tool).get("is_destructive", True)),
        )))

        try:
            response = await fut
        finally:
            self._pending.pop(req_id, None)

        choice = response.get("choice")
        feedback = str(response.get("feedback") or "")
        # Unknown scope → session; never silently elevate an unvalidated value.
        scope: Literal["project", "session"] = (
            "project" if response.get("scope") == "project" else "session"
        )
        if choice == "always":
            return AskerResponse(
                choice="always",
                scope=scope,
                rule=rule_hint,
                feedback=feedback,
            )
        if choice == "accept":
            return AskerResponse(choice="accept", feedback=feedback)
        return AskerResponse(choice="deny", feedback=feedback)

    def feed_response(self, payload: dict[str, Any]) -> bool:
        req_id = payload.get("id")
        if not isinstance(req_id, str):
            return False
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return False
        fut.set_result(payload)
        return True

    def deny_all_pending(self, *, feedback: str = "permission_request_cancelled") -> int:
        pending = [
            (req_id, fut)
            for req_id, fut in self._pending.items()
            if not fut.done()
        ]
        for req_id, fut in pending:
            fut.set_result({
                "id": req_id,
                "choice": "deny",
                "feedback": feedback,
            })
        return len(pending)


def feed_permission_response(
    *,
    asker: IpcAsker,
    payload: dict[str, Any],
    emit: EventEmitter,
) -> bool:
    if asker.feed_response(payload):
        return True
    emit({
        "event": "error",
        "message": f"no pending permission request for id={payload.get('id')!r}",
    })
    return False


async def run_session_driver(
    *,
    emit: EventEmitter,
    reader: RequestReader | None = None,
    load_config_fn: Any = load_config,
    make_model_for_spec_fn: Any = make_model_for_spec,
    make_permission_hook_fn: Any = make_permission_hook,
    agent_cls: type[Agent] = Agent,
    perm_store_module: Any = perm_store,
) -> int:
    cfg = load_config_fn()
    spec = cfg.router.get("default", "")
    if not spec:
        emit({
            "event": "error",
            "message": "config.router['default'] is missing — cannot start headless",
        })
        return 1

    model = make_model_for_spec_fn(spec, cfg)
    storage_path = Path(cfg.storage.path).expanduser()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = SessionStorage(storage_path)

    project_root = Path.cwd()
    try:
        perm_cfg = perm_store_module.load(project_root)
        known_tools = list(cfg.tools.enabled) + ["mcp__*"]
        disk_rules = perm_store_module.load_ruleset(project_root, known_tool_names=known_tools)
        deny_rules = perm_store_module.load_deny_ruleset(project_root)
        ask_rules = perm_store_module.load_ask_ruleset(project_root)
    except Exception as exc:  # noqa: BLE001
        emit({
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
    asker = IpcAsker(emit=emit)
    mode: Literal["default", "bypass", "plan", "accept_edits"] = perm_cfg.mode
    if mode == "bypass" and perm_cfg.disable_bypass:
        storage.close()
        emit({
            "event": "error",
            "message": "bypass mode is disabled by config (permissions.disable_bypass=true)",
        })
        return 1

    def _live_mode() -> Literal["default", "bypass", "plan", "accept_edits"]:
        return mode

    permission_hook = make_permission_hook_fn(
        asker=asker,
        session=session,
        rules=ruleset,
        project_root=project_root,
        mode=_live_mode,
        safety=safety_policy,
        deny_rules=deny_rules,
        ask_rules=ask_rules,
    )
    hooks = HookChain(pre_tool=[permission_hook])

    agent = agent_cls(
        config=cfg,
        model=model,
        storage=storage,
        hooks=hooks,
        session_rules=session,
        mode=mode,
        disable_bypass=perm_cfg.disable_bypass,
    )
    emit({"event": "ready", "session_id": agent.session_id, "model": spec})
    emit(dict(agent_state_to_wire(agent, 0.0)))

    turn_task: asyncio.Task[None] | None = None

    async def _drive_turn(text: str) -> None:
        try:
            async for event in stream_agent_wire(agent, text):
                emit(dict(event))
        except Exception as exc:  # noqa: BLE001
            emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    try:
        active_reader: RequestReader
        if reader is None:
            loop = asyncio.get_running_loop()
            stream_reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(stream_reader)
            await loop.connect_read_pipe(lambda: protocol, __import__("sys").stdin)
            active_reader = stream_reader
        else:
            active_reader = reader

        while True:
            line = await active_reader.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError as exc:
                emit({"event": "error", "message": f"bad request: {exc}"})
                continue

            kind = request.get("kind")
            if kind == "permission_response":
                feed_permission_response(asker=asker, payload=request, emit=emit)
                continue

            if kind == "prompt":
                text = request.get("text", "")
                if not isinstance(text, str) or not text:
                    emit({"event": "error", "message": "empty prompt"})
                    continue
                turn_task = asyncio.create_task(_drive_turn(text))
                while not turn_task.done():
                    try:
                        line = await asyncio.wait_for(active_reader.readline(), timeout=0.1)
                    except TimeoutError:
                        continue
                    if not line:
                        asker.deny_all_pending(feedback="stdin_closed")
                        break
                    try:
                        sub = json.loads(line.decode("utf-8").strip())
                    except json.JSONDecodeError as exc:
                        emit({"event": "error", "message": f"bad request mid-turn: {exc}"})
                        continue
                    if sub.get("kind") == "permission_response":
                        feed_permission_response(asker=asker, payload=sub, emit=emit)
                    else:
                        emit({
                            "event": "error",
                            "message": (
                                "only permission_response accepted mid-turn; "
                                f"got kind={sub.get('kind')!r}"
                            ),
                        })
                await turn_task
                turn_task = None
                continue

            emit({"event": "error", "message": f"unsupported request kind: {kind!r}"})
    finally:
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
        await agent.aclose()
        emit({"event": "exited"})
    return 0


__all__ = ["IpcAsker", "feed_permission_response", "run_session_driver"]
