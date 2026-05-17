"""Tests for Server-Sent Events framing."""

from __future__ import annotations

from importlib import import_module

import pytest

from aura.adapters.protocol.stream import encode_json_sse, encode_sse


def test_encode_sse_minimal_data_frame() -> None:
    assert encode_sse.__module__ == "aura.adapters.protocol.stream"
    assert encode_sse("hello") == "data: hello\n\n"


def test_legacy_transport_sse_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError, match=r"aura\.transport(\.sse)?"):
        import_module("aura.transport.sse")


def test_encode_sse_supports_event_id_retry_and_multiline_data() -> None:
    assert encode_sse(
        "hello\nworld",
        event="message",
        id="42",
        retry=1000,
    ) == "id: 42\nevent: message\nretry: 1000\ndata: hello\ndata: world\n\n"


def test_encode_json_sse_compacts_json_payload() -> None:
    assert encode_json_sse(
        {"type": "RUN_STARTED", "runId": "run-1"},
        event="agui",
    ) == 'event: agui\ndata: {"type":"RUN_STARTED","runId":"run-1"}\n\n'
