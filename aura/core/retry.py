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
    "509",  # Aliyun DashScope: server overloaded
    "timeout",
    "connection",
    "overloaded",
    "try again",
    # Provider-specific phrasings.
    "server is busy",  # DeepSeek "server is busy, please try again"
    "system busy",  # Zhipu GLM
    "service unavailable",
)

# Provider-specific structured error codes that map to "transient — retry".
# Mirrors the ``_CONTEXT_OVERFLOW_CODES`` pattern in ``agent.py``: substring
# match on the rendered exception text catches localized SDK messages
# where the prose has been translated but the code field stays stable.
_RETRIABLE_CODES: tuple[str, ...] = (
    # Zhipu GLM: 1001 = system busy, 1002 = rate limit exceeded.
    "1001",
    "1002",
    # Aliyun DashScope: 1003 = throttled. (1261 is overflow, NOT retriable.)
    "1003",
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
    raw = str(exc)
    msg = f" {raw.lower()} "
    if any(sub in msg for sub in _NON_RETRIABLE_SUBSTRINGS):
        return False
    if any(sub in msg for sub in _RETRIABLE_SUBSTRINGS):
        return True
    # Provider-specific structured codes — match against the un-spaced
    # rendered exception so quoted JSON-like fragments hit (`'code': '1001'`).
    for code in _RETRIABLE_CODES:
        if f"'code': '{code}'" in raw or f'"code": "{code}"' in raw:
            return True
    return False


# F-01-004 — server-suggested retry delay cap. Even if the provider says
# "wait 2 hours" we cap the honored value at 5 minutes so a misbehaving
# server can't park a turn forever; the surrounding ``max_attempts`` budget
# still bounds the total time.
_RETRY_AFTER_MAX_S: float = 300.0


def _extract_retry_after(exc: BaseException) -> float | None:
    """Pull a server-suggested retry delay (seconds) out of ``exc``, or None.

    Walks two surfaces in order:

    1. ``exc.response.headers["retry-after"]`` — HTTP standard. Value is
       either an integer-seconds string or an HTTP-date; we honor the
       seconds form and ignore the date form (rare in 429/503 responses
       and would require pulling email.utils into a hot path).
    2. SDK convenience fields on the exception itself — ``retry_after`` /
       ``retry_after_ms`` as set by openai / anthropic Python SDKs.

    Returns ``None`` when no usable hint is present, the value is
    malformed, or it parses as ``<= 0``. Returned value is capped at
    :data:`_RETRY_AFTER_MAX_S`.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw: object | None = None
        # ``headers`` may be a plain dict or a case-insensitive mapping;
        # try both lookup forms before giving up.
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
        except Exception as exc:
            # F-01-013 — narrowed from BaseException so KeyboardInterrupt /
            # SystemExit propagate to the caller instead of being classified
            # and (for KeyboardInterrupt's "interrupt" message) potentially
            # silently treated as non-retriable + re-raised after journaling.
            last_exc = exc
            if not retriable(exc):
                raise
            # Last attempt: no sleep, just re-raise after the for loop exits.
            if attempt == max_attempts - 1:
                break
            # F-01-004 — honor server-supplied Retry-After when present.
            # Falls through to exponential backoff otherwise.
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
            # asyncio.sleep raises CancelledError on task cancel — propagate,
            # do not swallow. That is the documented contract: users expect
            # Ctrl+C during backoff to tear down the turn instantly.
            await asyncio.sleep(wait)

    # Exhausted all attempts. ``last_exc`` is non-None because the loop only
    # exits here after at least one ``except`` branch ran.
    assert last_exc is not None
    raise last_exc
