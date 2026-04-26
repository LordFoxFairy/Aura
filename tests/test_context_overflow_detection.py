"""Provider-specific context-overflow signatures must trip reactive compact.

Bug pin: an actual user session hit ``BadRequestError: 400 - {'error':
{'code': '1261', 'message': 'Prompt exceeds max length'}}`` from Aliyun
DashScope and the reactive compact path did NOT fire because none of the
seeded phrases matched. Each known provider phrasing now has a regression
guard here so a future detector refactor cannot silently lose coverage.
"""

from __future__ import annotations

import pytest

from aura.core.agent import _is_context_overflow


class TestKnownPhrasings:
    """Each canonical provider error string must trip the detector."""

    @pytest.mark.parametrize("msg", [
        # OpenAI canonical.
        "This model's maximum context length is 8192 tokens. However, you requested 10000 tokens.",
        "Error code: 400 - context_length_exceeded",
        # Anthropic.
        "prompt is too long: 250000 tokens > 200000 maximum",
        # Aliyun DashScope — the user's actual reported case.
        (
            "BadRequestError: Error code: 400 - {'error': "
            "{'code': '1261', 'message': 'Prompt exceeds max length'}}"
        ),
        "Prompt exceeds max length",
        "Input too long for the model.",
        # Google Gemini.
        "Request payload size exceeds the limit: 32768",
        # Generic.
        "request had too many tokens",
        "max_tokens exceeded by 5000",
    ])
    def test_phrase_matches(self, msg: str) -> None:
        assert _is_context_overflow(RuntimeError(msg)) is True, msg


class TestStructuredCodeMatching:
    """Even if the message gets translated, the structured code stays."""

    def test_dashscope_1261_single_quote(self) -> None:
        # The OpenAI-compat client renders the body with single quotes —
        # the substring ``'code': '1261'`` is stable across SDK versions.
        msg = (
            "Error code: 400 - {'error': {'code': '1261', "
            "'message': 'limit excedido en chino', 'param': None}}"
        )
        assert _is_context_overflow(RuntimeError(msg)) is True

    def test_dashscope_1261_double_quote(self) -> None:
        msg = 'Error code: 400 - {"error": {"code": "1261", "message": "x"}}'
        assert _is_context_overflow(RuntimeError(msg)) is True


class TestNegatives:
    """Unrelated errors must NOT trip the detector — false positives would
    burn provider quota on doomed compact attempts every time the model
    surfaces any non-overflow error.
    """

    @pytest.mark.parametrize("msg", [
        "rate limit exceeded",
        "401 unauthorized",
        "Connection reset by peer",
        "JSON decode error",
        "Tool not found: bash_typo",
        "",
    ])
    def test_negative(self, msg: str) -> None:
        assert _is_context_overflow(RuntimeError(msg)) is False, msg
