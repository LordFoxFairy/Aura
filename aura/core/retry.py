"""Async retry helper with exponential backoff + jitter — transient-error
recovery for LLM provider calls.

Scope is narrow by design: wrap ONE awaitable, the ``model.ainvoke(...)`` site
in :class:`aura.core.loop.AgentLoop`. Tool retries are excluded because tool
semantics differ (some tools must not auto-retry — bash, write_file, etc.).

On a retriable exception we sleep ``min(base * 2**attempt + jitter, max)`` then
try again up to ``max_attempts`` total. Non-retriable exceptions — including
:class:`asyncio.CancelledError` — propagate immediately. ``CancelledError``
raised during the backoff sleep also propagates, honoring Ctrl+C cleanly.

Each retry emits a ``llm_retry`` journal event so operators can see transient
failure patterns in the audit log without enabling debug logs.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aura.core.persistence import journal

T = TypeVar("T")

# Exception class names that are ALWAYS retriable. Matched by class name
# (not isinstance) so Aura doesn't need to import openai / anthropic SDKs —
# the SDK surface shifts between versions and across providers. Covers the
# OpenAI and Anthropic Python SDKs; new provider SDKs get added here.
_RETRIABLE_CLASS_NAMES: frozenset[str] = frozenset({
    "RateLimitError",
    "ServiceUnavailableError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ConflictError",
})

# Exception class names that are NEVER retriable — user errors / auth errors.
# Takes precedence over substring matching because an auth-error message can
# include "timeout" ("request timed out waiting for authentication") and
# we would retry forever.
_NON_RETRIABLE_CLASS_NAMES: frozenset[str] = frozenset({
    "AuthenticationError",
    "BadRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnprocessableEntityError",
})

# Substrings that signal a transient condition when present in the exception
# message. Lowercased before compare. Covers the common HTTP status codes
# providers surface via plain strings when the SDK is not available (e.g.
# direct requests / aiohttp). "overloaded" is Anthropic's specific phrasing.
_RETRIABLE_SUBSTRINGS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "429",
    "502",
    "503",
    "504",
    "timeout",
    "connection",
    "overloaded",
    "try again",
)

# Substrings that signal a PERMANENT failure — wins over retriable matches.
# ``"401"`` etc. can collide with a retriable message that happens to mention
# a port or date, so we keep this list tight.
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
    """Classify ``exc`` as retriable.

    Precedence: CancelledError never retries -> explicit class-name deny-list
    -> explicit class-name allow-list -> message deny-list -> message allow-list
    -> default deny. Being conservative by default means unknown exceptions
    surface to the caller instead of retrying silently.
    """
    if isinstance(exc, asyncio.CancelledError):
        return False
    cls_name = type(exc).__name__
    if cls_name in _NON_RETRIABLE_CLASS_NAMES:
        return False
    if cls_name in _RETRIABLE_CLASS_NAMES:
        return True
    # Surround the message with spaces so status-code substrings like " 401 "
    # match at message boundaries without false-positiving inside URLs or
    # timestamps ("2024-04-01T00:00:00").
    msg = f" {str(exc).lower()} "
    if any(sub in msg for sub in _NON_RETRIABLE_SUBSTRINGS):
        return False
    return any(sub in msg for sub in _RETRIABLE_SUBSTRINGS)


def _compute_delay(
    attempt: int, *, base: float, cap: float, jitter: bool,
) -> float:
    """Exponential backoff with optional jitter. ``attempt`` is 0-indexed.

    With jitter off the sequence is base, 2*base, 4*base, ... (capped at
    ``cap``) — deterministic, test-friendly. Jitter adds up to 0.5s of
    uniform noise so a cluster of retrying clients does not reconverge on
    the same retry instant (thundering-herd).
    """
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
    """Call ``fn()`` with exponential-backoff retry on transient errors.

    Parameters mirror :class:`aura.config.schema.RetryConfig`. The callable
    is wrapped in a zero-arg lambda at the call site so we can re-invoke it
    without threading kwargs through.

    ``max_attempts=1`` disables retries (one try, one re-raise) — useful for
    tests and as a kill switch in config. The loop re-raises the LAST
    exception; caller never sees a partial result on failure.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except asyncio.CancelledError:
            # Cancellation ALWAYS propagates — Ctrl+C / task.cancel() must
            # tear down promptly. Do not classify, do not journal, do not sleep.
            raise
        except BaseException as exc:
            last_exc = exc
            if not retriable(exc):
                raise
            # Last attempt: no sleep, just re-raise after the for loop exits.
            if attempt == max_attempts - 1:
                break
            wait = _compute_delay(
                attempt, base=base_delay_s, cap=max_delay_s, jitter=jitter,
            )
            journal.write(
                "llm_retry",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                wait_seconds=round(wait, 3),
                reason=type(exc).__name__,
            )
            # asyncio.sleep raises CancelledError on task cancel — propagate,
            # do not swallow. That is the documented contract: users expect
            # Ctrl+C during backoff to tear down the turn instantly.
            await asyncio.sleep(wait)

    # Exhausted all attempts. ``last_exc`` is non-None because the loop only
    # exits here after at least one ``except`` branch ran.
    assert last_exc is not None
    raise last_exc
