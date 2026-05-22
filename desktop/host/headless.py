"""Headless SSE wrapper for the shared desktop session service.

This module is desktop-specific only at the stdio boundary: it reads NDJSON
requests from stdin, writes SSE-framed events to stdout, and delegates session
behavior to ``desktop.host.session_service``. It intentionally preserves a few
legacy helper names for tests while the runtime ownership has moved.
"""

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
from aura.infrastructure.wire.stream import encode_sse
from aura.infrastructure.wire.wire import agent_state_to_wire, event_to_wire
from desktop.host import session_service


class IpcAsker(session_service.IpcAsker):
    """Legacy zero-arg wrapper over the shared session-service asker."""

    def __init__(self) -> None:
        super().__init__(emit=_emit)


def _emit(payload: dict[str, Any]) -> None:
    """Write one SSE frame to stdout + flush."""
    sys.stdout.write(encode_sse(cast(Any, payload)))
    sys.stdout.flush()


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Compatibility wrapper for the shared Aura wire serializer."""
    return dict(event_to_wire(event))


def _build_aura_state(agent: Any, last_turn_seconds: float) -> dict[str, Any]:
    """Compatibility wrapper for the shared Aura state serializer."""
    return dict(agent_state_to_wire(agent, last_turn_seconds))


def _feed_permission_response(asker: IpcAsker, payload: dict[str, Any]) -> bool:
    """Legacy local wrapper over the shared feed helper."""
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
    """Entry point — bootstrap the shared session driver over stdio."""
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        _emit({"event": "exited"})
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
