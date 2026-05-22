"""Async retry with exponential backoff + jitter for transient LLM errors.

Wraps one awaitable (the ``model.ainvoke`` site in :class:`AgentLoop`).
Tool retries are intentionally excluded.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aura.infrastructure.persistence import journal

T = TypeVar("T")

# Matched by class name (no SDK imports). Covers openai + anthropic.
_RETRIABLE_CLASS_NAMES: frozenset[str] = frozenset({
    "RateLimitError",
    "ServiceUnavailableError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ConflictError",
})

# Takes precedence over substring matching — an auth error message can
# include "timeout" ("request timed out waiting for authentication").
_NON_RETRIABLE_CLASS_NAMES: frozenset[str] = frozenset({
    "AuthenticationError",
    "BadRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnprocessableEntityError",
})

_RETRIABLE_SUBSTRINGS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "429",
    "502",
    "503",
    "504",
    "509",  # DashScope server overloaded
    "timeout",
    "connection",
    "overloaded",
    "try again",
    "server is busy",  # DeepSeek
    "system busy",  # GLM
    "service unavailable",
)

# Substring-match on rendered exception text — catches localized SDK
# messages where prose is translated but the code field stays stable.
_RETRIABLE_CODES: tuple[str, ...] = (
    "1001",  # GLM system busy
    "1002",  # GLM rate limit
    "1003",  # DashScope throttled (1261 is overflow, NOT retriable)
)

# Tight permanent-failure list — wins over retriable matches.
_NON_RETRIABLE_SUBSTRINGS: tuple[str, ...] = (
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    " 400 ",
    " 401 ",
    " 403 ",
    " 404 ",
)


def _is_retriable(exc: BaseException) -> bool:
    """CancelledError never retries → deny-list → allow-list → msg-deny → msg-allow → False."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    cls_name = type(exc).__name__
    if cls_name in _NON_RETRIABLE_CLASS_NAMES:
        return False
    if cls_name in _RETRIABLE_CLASS_NAMES:
        return True
    # Surround with spaces so " 401 " matches at boundaries.
    raw = str(exc)
    msg = f" {raw.lower()} "
    if any(sub in msg for sub in _NON_RETRIABLE_SUBSTRINGS):
        return False
    if any(sub in msg for sub in _RETRIABLE_SUBSTRINGS):
        return True
    for code in _RETRIABLE_CODES:
        if f"'code': '{code}'" in raw or f'"code": "{code}"' in raw:
            return True
    return False


# Even "wait 2 hours" is capped at 5 min so a misbehaving server can't park forever.
_RETRY_AFTER_MAX_S: float = 300.0


def _extract_retry_after(exc: BaseException) -> float | None:
    """Server-suggested delay (seconds) from ``Retry-After`` header or SDK fields."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw: object | None = None
        try:
            raw = headers.get("retry-after")
            if raw is None:
                raw = headers.get("Retry-After")
        except (AttributeError, TypeError):
            raw = None
        if raw is not None:
            try:
                seconds = float(str(raw).strip())
            except (TypeError, ValueError):
                seconds = -1.0
            if seconds > 0:
                return min(seconds, _RETRY_AFTER_MAX_S)

    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, (int, float)) and direct > 0:
        return min(float(direct), _RETRY_AFTER_MAX_S)

    direct_ms = getattr(exc, "retry_after_ms", None)
    if isinstance(direct_ms, (int, float)) and direct_ms > 0:
        return min(float(direct_ms) / 1000.0, _RETRY_AFTER_MAX_S)

    return None


def _compute_delay(
    attempt: int, *, base: float, cap: float, jitter: bool,
) -> float:
    """Exponential backoff. ``attempt`` is 0-indexed; jitter adds ≤0.5s noise."""
    delay: float = base * (2 ** attempt)
    if jitter:
        delay += random.uniform(0, 0.5)
    return float(min(delay, cap))


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    max_delay_s: float = 30.0,
    jitter: bool = True,
    retriable: Callable[[BaseException], bool] = _is_retriable,
) -> T:
    """Call ``fn()`` with backoff on transient errors.

    Cancellation always propagates. Non-retriable errors propagate
    immediately. ``max_attempts=1`` disables retries.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_exc = exc
            if not retriable(exc):
                raise
            if attempt == max_attempts - 1:
                break
            header_wait = _extract_retry_after(exc)
            wait = (
                header_wait
                if header_wait is not None
                else _compute_delay(
                    attempt, base=base_delay_s, cap=max_delay_s, jitter=jitter,
                )
            )
            journal.write(
                "llm_retry",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                wait_seconds=round(wait, 3),
                reason=type(exc).__name__,
                retry_after_source="header" if header_wait is not None else "backoff",
            )
            # asyncio.sleep raises CancelledError on task cancel — propagate.
            await asyncio.sleep(wait)

    assert last_exc is not None
    raise last_exc
