"""Provider-specific retry shapes — DashScope / Zhipu / DeepSeek / Moonshot.

Same pattern as ``test_context_overflow_detection.py`` for the retry
classifier: each provider's transient-overload signature must trip
``_is_retriable`` so the retry policy actually fires instead of bailing
on first hit. Audit gap caught by the post-v0.18.0 5-angle review.
"""

from __future__ import annotations

import pytest

from aura.core.retry import _is_retriable


class TestProviderTransientCodes:

    @pytest.mark.parametrize("msg", [
        # Aliyun DashScope 509 (overloaded).
        "Error code: 509 - server overloaded",
        # DeepSeek's literal phrasing.
        "server is busy, please try again later",
        # Zhipu GLM 1001 (system busy) / 1002 (rate limit).
        "Error: {'error': {'code': '1001', 'message': 'system busy'}}",
        "BadRequestError: {'error': {'code': '1002'}}",
        # DashScope 1003 throttle.
        '{"error": {"code": "1003", "message": "throttled"}}',
    ])
    def test_provider_transient_retriable(self, msg: str) -> None:
        assert _is_retriable(RuntimeError(msg)) is True, msg


class TestNonRetriableProviderCodes:

    @pytest.mark.parametrize("msg", [
        # 1261 is overflow — NOT retriable, the reactive-compact path
        # owns it. Pinning that explicitly here so a future refactor
        # doesn't accidentally lump it in with the transient codes.
        "Error code: 400 - {'error': {'code': '1261', 'message': 'Prompt exceeds max length'}}",
    ])
    def test_overflow_not_retriable(self, msg: str) -> None:
        assert _is_retriable(RuntimeError(msg)) is False, msg


class TestNegatives:

    @pytest.mark.parametrize("msg", [
        "401 unauthorized",
        "404 not found",
        "JSON parse error",
        "Tool unknown: bash_typo",
    ])
    def test_unrelated_not_retriable(self, msg: str) -> None:
        assert _is_retriable(RuntimeError(msg)) is False, msg
