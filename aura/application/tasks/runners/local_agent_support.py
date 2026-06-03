"""Runtime support for the local-agent runner: timeout resolution + token observer."""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, BaseMessage

from aura.application.hooks.protocols import PostModelHook
from aura.application.loop_state import LoopState
from aura.application.tasks.store import TasksStore
from aura.infrastructure.persistence import journal

# 5 minute defense-in-depth ceiling; ``AURA_SUBAGENT_TIMEOUT_SEC<=0`` disables.
DEFAULT_SUBAGENT_TIMEOUT_SEC: float = 300.0
_TIMEOUT_ENV_VAR = "AURA_SUBAGENT_TIMEOUT_SEC"


def resolve_timeout(override: float | None) -> float | None:
    """Pick the effective wallclock timeout (None == disabled).

    Precedence: explicit override > env var > default. ``<= 0`` flows through as ``None``.
    Malformed env values journal + fall through to the default.
    """
    if override is not None:
        return override if override > 0 else None
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is not None:
        try:
            parsed = float(raw)
        except ValueError:
            journal.write(
                "subagent_timeout_env_invalid",
                var=_TIMEOUT_ENV_VAR,
                value=raw,
            )
        else:
            return parsed if parsed > 0 else None
    return DEFAULT_SUBAGENT_TIMEOUT_SEC


def make_token_observer(store: TasksStore, task_id: str) -> PostModelHook:
    """post_model hook forwarding ``usage_metadata`` into the store; failures journaled."""
    async def _observe(
        *,
        ai_message: AIMessage,
        history: list[BaseMessage],  # noqa: ARG001 - protocol compliance
        state: LoopState,  # noqa: ARG001 - protocol compliance
        **_: object,
    ) -> None:
        try:
            usage = ai_message.usage_metadata
            if not usage:
                return
            in_t = int(usage.get("input_tokens", 0) or 0)
            out_t = int(usage.get("output_tokens", 0) or 0)
            store.record_token_usage(
                task_id, input_tokens=in_t, output_tokens=out_t,
            )
        except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            journal.write(
                "subagent_token_observer_error",
                task_id=task_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    return _observe


__all__ = [
    "DEFAULT_SUBAGENT_TIMEOUT_SEC",
    "make_token_observer",
    "resolve_timeout",
]
