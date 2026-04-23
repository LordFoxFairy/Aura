"""Tests for aura.core.retry — async retry helper with exponential backoff."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from aura.config.schema import RetryConfig
from aura.core.persistence import journal
from aura.core.retry import _is_retriable, with_retry

# --- helpers ---------------------------------------------------------------


class _FakeRateLimitError(Exception):
    """Stand-in for openai.RateLimitError without importing the SDK.

    We match by class name so this needs the same string — the retry helper
    does ``type(exc).__name__``, not ``isinstance(exc, …)``.
    """


_FakeRateLimitError.__name__ = "RateLimitError"


class _FakeAuthError(Exception):
    pass


_FakeAuthError.__name__ = "AuthenticationError"


def _make_counting_fn(
    *, errors: list[BaseException], final_value: Any = "ok",
) -> tuple[Any, list[int]]:
    """Factory: returns an async fn that raises from ``errors`` in order,
    then returns ``final_value``. Also returns a counter list the caller
    can read — element 0 is the call count."""

    calls = [0]

    async def fn() -> Any:
        calls[0] += 1
        if errors:
            exc = errors.pop(0)
            raise exc
        return final_value

    return fn, calls


# --- _is_retriable classification -----------------------------------------


def test_is_retriable_class_name_rate_limit() -> None:
    assert _is_retriable(_FakeRateLimitError("slow down")) is True


def test_is_retriable_auth_error_not_retriable() -> None:
    assert _is_retriable(_FakeAuthError("401 unauthorized")) is False


def test_is_retriable_429_in_message() -> None:
    assert _is_retriable(RuntimeError("HTTP 429 too many requests")) is True


def test_is_retriable_503_in_message() -> None:
    assert _is_retriable(RuntimeError("HTTP 503 service unavailable")) is True


def test_is_retriable_401_in_message_not_retriable() -> None:
    assert _is_retriable(RuntimeError("HTTP 401 invalid api key")) is False


def test_is_retriable_cancelled_error_not_retriable() -> None:
    assert _is_retriable(asyncio.CancelledError()) is False


def test_is_retriable_unknown_exception_defaults_to_false() -> None:
    # Conservative default: unknown = don't retry, surface to caller.
    assert _is_retriable(ValueError("garbled response")) is False


def test_is_retriable_overloaded_substring() -> None:
    # Anthropic-specific phrasing.
    assert _is_retriable(RuntimeError("Overloaded: try again shortly")) is True


# --- with_retry success / failure paths -----------------------------------


@pytest.mark.asyncio
async def test_succeeds_on_first_try_called_once() -> None:
    fn, calls = _make_counting_fn(errors=[], final_value="done")
    result = await with_retry(fn, max_attempts=3, jitter=False, base_delay_s=0.001)
    assert result == "done"
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_succeeds_on_second_try_journal_has_one_retry(
    tmp_path: Path,
) -> None:
    journal.configure(tmp_path / "j.jsonl")
    try:
        fn, calls = _make_counting_fn(
            errors=[_FakeRateLimitError("429")], final_value=42,
        )
        result = await with_retry(
            fn, max_attempts=3, jitter=False, base_delay_s=0.001, max_delay_s=0.01,
        )
        assert result == 42
        assert calls[0] == 2
        # Exactly one llm_retry event for the single retry.
        lines = (tmp_path / "j.jsonl").read_text().splitlines()
        retries = [json.loads(line) for line in lines if '"llm_retry"' in line]
        assert len(retries) == 1
        assert retries[0]["attempt"] == 1
        assert retries[0]["reason"] == "RateLimitError"
        assert retries[0]["wait_seconds"] >= 0
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_fails_all_attempts_reraises_last_exception() -> None:
    # Three retriable errors; max_attempts=3 → all attempts burn, last raises.
    errs: list[BaseException] = [
        _FakeRateLimitError("one"),
        _FakeRateLimitError("two"),
        _FakeRateLimitError("three"),
    ]
    fn, calls = _make_counting_fn(errors=errs)
    with pytest.raises(_FakeRateLimitError, match="three"):
        await with_retry(
            fn, max_attempts=3, jitter=False, base_delay_s=0.001, max_delay_s=0.01,
        )
    assert calls[0] == 3


@pytest.mark.asyncio
async def test_non_retriable_exception_reraises_immediately_no_retries() -> None:
    fn, calls = _make_counting_fn(errors=[_FakeAuthError("401")])
    with pytest.raises(_FakeAuthError):
        await with_retry(
            fn, max_attempts=5, jitter=False, base_delay_s=0.001,
        )
    # Single call — no retries despite max_attempts=5.
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_max_attempts_one_disables_retries() -> None:
    fn, calls = _make_counting_fn(errors=[_FakeRateLimitError("429")])
    with pytest.raises(_FakeRateLimitError):
        await with_retry(fn, max_attempts=1, jitter=False, base_delay_s=0.001)
    assert calls[0] == 1


# --- backoff math ----------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_backoff_without_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With jitter off, attempt 0 → 1.0s, attempt 1 → 2.0s, attempt 2 → 4.0s.
    # We capture the sleeps instead of actually waiting to keep the test
    # fast and deterministic.
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    fn, _ = _make_counting_fn(errors=[
        _FakeRateLimitError("a"),
        _FakeRateLimitError("b"),
        _FakeRateLimitError("c"),
        _FakeRateLimitError("d"),
    ])
    with pytest.raises(_FakeRateLimitError):
        await with_retry(
            fn, max_attempts=4, jitter=False, base_delay_s=1.0, max_delay_s=100.0,
        )
    # 3 sleeps (between the 4 attempts): 1.0, 2.0, 4.0.
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_max_delay_caps_very_high_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(sec: float) -> None:
        sleeps.append(sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # attempt 0: 1.0  (capped ok)
    # attempt 1: 2.0  (capped ok)
    # attempt 2: 4.0  (capped ok)
    # attempt 3: 8.0  (capped → 5.0)
    # attempt 4: 16.0 (capped → 5.0)
    fn, _ = _make_counting_fn(errors=[_FakeRateLimitError(str(i)) for i in range(6)])
    with pytest.raises(_FakeRateLimitError):
        await with_retry(
            fn, max_attempts=6, jitter=False, base_delay_s=1.0, max_delay_s=5.0,
        )
    assert sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]


# --- cancellation semantics ----------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_error_during_backoff_propagates() -> None:
    # asyncio.sleep raising CancelledError must tear down the retry loop
    # instantly — Ctrl+C during a 30s wait is unacceptable UX if swallowed.
    fn, calls = _make_counting_fn(errors=[
        _FakeRateLimitError("first"),
    ])

    async def run() -> None:
        await with_retry(
            fn, max_attempts=3, jitter=False, base_delay_s=60.0, max_delay_s=120.0,
        )

    task = asyncio.create_task(run())
    # Give the task a tick to hit the first retry and start sleeping.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The fn ran once, raised, we got into the sleep, then cancellation.
    assert calls[0] == 1


@pytest.mark.asyncio
async def test_cancelled_error_raised_by_fn_propagates() -> None:
    # Even if fn() itself raises CancelledError (e.g. aiohttp cancellation
    # bubbling up), we must NOT treat it as retriable.
    fn, calls = _make_counting_fn(errors=[asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await with_retry(fn, max_attempts=3, jitter=False, base_delay_s=0.001)
    assert calls[0] == 1


# --- journal events --------------------------------------------------------


@pytest.mark.asyncio
async def test_journal_events_record_attempt_wait_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal.configure(tmp_path / "j.jsonl")

    async def fake_sleep(_sec: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    try:
        fn, _ = _make_counting_fn(errors=[
            _FakeRateLimitError("a"),
            _FakeRateLimitError("b"),
        ])
        await with_retry(
            fn, max_attempts=5, jitter=False, base_delay_s=1.0, max_delay_s=4.0,
        )
        lines = (tmp_path / "j.jsonl").read_text().splitlines()
        retries = [json.loads(line) for line in lines if '"llm_retry"' in line]
        assert len(retries) == 2
        assert retries[0]["attempt"] == 1
        assert retries[0]["wait_seconds"] == 1.0
        assert retries[0]["reason"] == "RateLimitError"
        assert retries[1]["attempt"] == 2
        assert retries[1]["wait_seconds"] == 2.0
        assert retries[1]["max_attempts"] == 5
    finally:
        journal.reset()


@pytest.mark.asyncio
async def test_invalid_max_attempts_raises() -> None:
    async def fn() -> str:
        return "never"

    with pytest.raises(ValueError, match="max_attempts"):
        await with_retry(fn, max_attempts=0)


# --- RetryConfig schema ---------------------------------------------------


def test_retry_config_defaults() -> None:
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.base_delay_s == 1.0
    assert cfg.max_delay_s == 30.0


def test_retry_config_bounds_enforced() -> None:
    from pydantic import ValidationError

    # max_attempts must be in [1, 10].
    with pytest.raises(ValidationError):
        RetryConfig(max_attempts=0)
    with pytest.raises(ValidationError):
        RetryConfig(max_attempts=11)
    # base_delay_s / max_delay_s must be > 0.
    with pytest.raises(ValidationError):
        RetryConfig(base_delay_s=0)
    with pytest.raises(ValidationError):
        RetryConfig(max_delay_s=-1.0)


def test_retry_config_extra_forbid() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RetryConfig.model_validate({"max_attempts": 3, "bogus": 1})


# --- custom retriable callable -------------------------------------------


@pytest.mark.asyncio
async def test_custom_retriable_callable_overrides_default() -> None:
    # Accept every exception as retriable (custom) even though ValueError
    # would NOT retry by default.
    fn, calls = _make_counting_fn(errors=[
        ValueError("normally not retriable"),
    ], final_value="ok")
    result = await with_retry(
        fn,
        max_attempts=3,
        jitter=False,
        base_delay_s=0.001,
        max_delay_s=0.01,
        retriable=lambda _: True,
    )
    assert result == "ok"
    assert calls[0] == 2
