"""Headless SSE wrapper around ``desktop.host.session_service``."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from typing import Any, cast

from aura.application.hooks.permission import make_permission_hook
from aura.config.loader import load_config
from aura.core.agent import Agent
from aura.infrastructure import permission_store as perm_store
from aura.infrastructure.llm import make_model_for_spec
from aura.infrastructure.wire.serialize import agent_state_to_wire, event_to_wire
from aura.infrastructure.wire.stream import encode_sse
from desktop.host import session_service


class IpcAsker(session_service.IpcAsker):
    def __init__(self) -> None:
        super().__init__(emit=_emit)


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(encode_sse(cast(Any, payload)))
    sys.stdout.flush()


def _event_to_dict(event: Any) -> dict[str, Any]:
    return dict(event_to_wire(event))


def _build_aura_state(agent: Any, last_turn_seconds: float) -> dict[str, Any]:
    return dict(agent_state_to_wire(agent, last_turn_seconds))


def _feed_permission_response(asker: IpcAsker, payload: dict[str, Any]) -> bool:
    return session_service.feed_permission_response(
        asker=asker,
        payload=payload,
        emit=_emit,
    )


async def _run() -> int:
    try:
        return await session_service.run_session_driver(
            emit=_emit,
            load_config_fn=load_config,
            make_model_for_spec_fn=make_model_for_spec,
            make_permission_hook_fn=make_permission_hook,
            agent_cls=Agent,
            perm_store_module=perm_store,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'emit'" not in str(exc):
            raise
        legacy_runner = cast(Callable[[], Awaitable[int]], session_service.run_session_driver)
        return await legacy_runner()


def main() -> int:
    # ``exited`` is emitted by run_session_driver's finally; do not re-emit here.
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


__all__ = [
    "IpcAsker",
    "_build_aura_state",
    "_emit",
    "_event_to_dict",
    "_feed_permission_response",
    "_run",
    "main",
    "session_service",
]


if __name__ == "__main__":
    sys.exit(main())
