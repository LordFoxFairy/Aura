"""Transport-neutral session driver for desktop front-end wrappers."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import sys
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from aura.application.hooks import HookChain
from aura.application.hooks.permission import make_permission_hook
from aura.application.hooks.protocols import PreToolHook
from aura.application.permission.asker import AskerResponse
from aura.application.session import AgentSession
from aura.config.loader import load_config
from aura.config.schema import AuraConfig, PermissionsConfig
from aura.domain.permission.defaults import DEFAULT_ALLOW_RULES
from aura.domain.permission.mode import Mode
from aura.domain.permission.rule import Rule
from aura.domain.permission.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
    SafetyPolicy,
)
from aura.domain.permission.session import RuleSet, SessionRuleSet
from aura.domain.tool_meta_access import meta_dict
from aura.infrastructure import permission_store as perm_store
from aura.infrastructure.llm import make_model_for_spec
from aura.infrastructure.persistence.storage import SessionStorage
from aura.infrastructure.wire.serialize import agent_state_to_wire, permission_request_to_wire
from aura.infrastructure.wire.stream import stream_agent_wire

_KNOWN_KINDS = ("prompt", "permission_response")


class _PromptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["prompt"]
    # Non-str/missing text coerces to "" so the empty-prompt guard owns the reject.
    text: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return value if isinstance(value, str) else ""


class _PermissionResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["permission_response"]
    id: str
    choice: str = "deny"
    feedback: str = ""
    scope: str | None = None


_InboundRequest = Annotated[
    _PromptRequest | _PermissionResponseRequest,
    Field(discriminator="kind"),
]
_INBOUND_ADAPTER: TypeAdapter[_PromptRequest | _PermissionResponseRequest] = TypeAdapter(
    _InboundRequest,
)


class _RejectReason(Enum):
    DECODE = "decode"  # malformed JSON or shape that fails Pydantic validation
    UNKNOWN_KIND = "unknown_kind"  # valid object, kind absent or not a known tag


@dataclasses.dataclass(frozen=True, slots=True)
class _Rejected:
    reason: _RejectReason
    detail: str  # decode → exception text; unknown_kind → repr(kind)


def _parse_inbound(
    line: bytes,
) -> _PromptRequest | _PermissionResponseRequest | _Rejected:
    try:
        raw = json.loads(line.decode("utf-8").strip())
    except json.JSONDecodeError as exc:
        return _Rejected(_RejectReason.DECODE, str(exc))
    kind = raw.get("kind") if isinstance(raw, dict) else None
    if kind not in _KNOWN_KINDS:
        return _Rejected(_RejectReason.UNKNOWN_KIND, repr(kind))
    try:
        return _INBOUND_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        return _Rejected(_RejectReason.DECODE, str(exc))


class EventEmitter(Protocol):
    def __call__(self, payload: dict[str, Any]) -> None: ...


class RequestReader(Protocol):
    async def readline(self) -> bytes: ...


class PermStoreModule(Protocol):
    def load(self, project_root: Path) -> PermissionsConfig: ...
    def load_ruleset(
        self, project_root: Path, *, known_tool_names: Iterable[str] | None = None,
    ) -> RuleSet: ...
    def load_deny_ruleset(self, project_root: Path) -> RuleSet: ...
    def load_ask_ruleset(self, project_root: Path) -> RuleSet: ...


class IpcAsker:
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
    load_config_fn: Callable[[], AuraConfig] = load_config,
    make_model_for_spec_fn: Callable[[str, AuraConfig], BaseChatModel] = make_model_for_spec,
    make_permission_hook_fn: Callable[..., PreToolHook] = make_permission_hook,
    agent_cls: type[AgentSession] = AgentSession,
    perm_store_module: PermStoreModule = perm_store,
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
    except Exception as exc:  # noqa: BLE001  # corrupt input falls back to default
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
    mode: Mode = perm_cfg.mode
    if mode == "bypass" and perm_cfg.disable_bypass:
        storage.close()
        emit({
            "event": "error",
            "message": "bypass mode is disabled by config (permissions.disable_bypass=true)",
        })
        return 1

    def _live_mode() -> Mode:
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
        except Exception as exc:  # noqa: BLE001  # fan-out callback must not poison loop
            emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})

    try:
        active_reader: RequestReader
        if reader is None:
            loop = asyncio.get_running_loop()
            stream_reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(stream_reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)
            active_reader = stream_reader
        else:
            active_reader = reader

        while True:
            line = await active_reader.readline()
            if not line:
                break
            req = _parse_inbound(line)
            if isinstance(req, _Rejected):
                if req.reason is _RejectReason.UNKNOWN_KIND:
                    emit({
                        "event": "error",
                        "message": f"unsupported request kind: {req.detail}",
                    })
                else:
                    emit({"event": "error", "message": f"bad request: {req.detail}"})
                continue

            if isinstance(req, _PermissionResponseRequest):
                feed_permission_response(asker=asker, payload=req.model_dump(), emit=emit)
                continue

            if not req.text:
                emit({"event": "error", "message": "empty prompt"})
                continue
            turn_task = asyncio.create_task(_drive_turn(req.text))
            while not turn_task.done():
                try:
                    line = await asyncio.wait_for(active_reader.readline(), timeout=0.1)
                except TimeoutError:
                    continue
                if not line:
                    asker.deny_all_pending(feedback="stdin_closed")
                    break
                sub = _parse_inbound(line)
                if isinstance(sub, _Rejected):
                    if sub.reason is _RejectReason.DECODE:
                        emit({
                            "event": "error",
                            "message": f"bad request mid-turn: {sub.detail}",
                        })
                    else:
                        emit({
                            "event": "error",
                            "message": (
                                "only permission_response accepted mid-turn; "
                                f"got kind={sub.detail}"
                            ),
                        })
                elif isinstance(sub, _PermissionResponseRequest):
                    feed_permission_response(asker=asker, payload=sub.model_dump(), emit=emit)
                else:
                    emit({
                        "event": "error",
                        "message": (
                            "only permission_response accepted mid-turn; "
                            "got kind='prompt'"
                        ),
                    })
            await turn_task
            turn_task = None
    finally:
        # Await after cancel so ``final`` flushes before ``exited`` on the wire.
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
            with contextlib.suppress(BaseException):
                await turn_task
        await agent.aclose()
        emit({"event": "exited"})
    return 0


__all__ = ["IpcAsker", "feed_permission_response", "run_session_driver"]
