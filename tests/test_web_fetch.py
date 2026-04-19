"""Tests for aura.tools.web_fetch."""

from __future__ import annotations

import importlib
import sys

import pytest

from aura.schemas.tool import ToolError
from aura.tools.web_fetch import WebFetchParams, _fetch

_wf_mod = importlib.import_module("aura.tools.web_fetch")
assert isinstance(_wf_mod, type(sys)), "expected module"


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/plain") -> None:
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            return self._body
        return self._body[:n]


def test_web_fetch_rejects_non_http_url() -> None:
    with pytest.raises(ToolError, match="http"):
        _fetch(url="file:///etc/passwd")


def _allow_all_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass the SSRF guard for tests that mock urlopen."""
    monkeypatch.setattr(_wf_mod, "_reject_private_host", lambda _host: None)


def test_web_fetch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_all_hosts(monkeypatch)
    fake_body = b"hello world"

    def _fake_urlopen(req: object, timeout: object) -> _FakeResponse:
        return _FakeResponse(fake_body, status=200, content_type="text/plain")

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    out = _fetch(url="https://example.com")
    assert out["status"] == 200
    assert out["content"] == "hello world"
    assert out["truncated"] is False


def test_web_fetch_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_all_hosts(monkeypatch)
    large_body = b"x" * (2 * 1024 * 1024)

    def _fake_urlopen(req: object, timeout: object) -> _FakeResponse:
        return _FakeResponse(large_body, status=200)

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    out = _fetch(url="https://example.com")
    assert out["truncated"] is True
    assert len(out["content"]) == 1024 * 1024


def test_web_fetch_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_all_hosts(monkeypatch)
    from urllib.error import URLError

    def _fake_urlopen(req: object, timeout: object) -> _FakeResponse:
        raise URLError("connection refused")

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    with pytest.raises(ToolError, match="fetch failed"):
        _fetch(url="https://example.com")


def test_web_fetch_rejects_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _wf_mod.socket,
        "getaddrinfo",
        lambda host, _port: [(0, 0, 0, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(ToolError, match="non-public IP"):
        _fetch(url="http://localhost")


def test_web_fetch_rejects_private_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _wf_mod.socket,
        "getaddrinfo",
        lambda host, _port: [(0, 0, 0, "", ("10.0.0.5", 0))],
    )
    with pytest.raises(ToolError, match="non-public IP"):
        _fetch(url="http://internal.corp")


def test_web_fetch_rejects_cloud_metadata_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    # 169.254.169.254 = AWS / GCP / Azure instance metadata endpoint.
    monkeypatch.setattr(
        _wf_mod.socket,
        "getaddrinfo",
        lambda host, _port: [(0, 0, 0, "", ("169.254.169.254", 0))],
    )
    with pytest.raises(ToolError, match="non-public IP"):
        _fetch(url="http://metadata.internal")


def test_web_fetch_rejects_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket as sk

    def _boom(host: str, _port: object) -> object:
        raise sk.gaierror("no such host")

    monkeypatch.setattr(_wf_mod.socket, "getaddrinfo", _boom)
    with pytest.raises(ToolError, match="dns resolve failed"):
        _fetch(url="http://nonexistent.invalid")


def test_web_fetch_rejects_malformed_url() -> None:
    with pytest.raises(ToolError, match="malformed URL"):
        _fetch(url="http://")


def test_web_fetch_capability_flags() -> None:
    from aura.tools.web_fetch import web_fetch

    meta = web_fetch.metadata or {}
    assert meta.get("is_read_only") is True
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is True


def test_web_fetch_timeout_bounds() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WebFetchParams(url="https://example.com", timeout=121)
    with pytest.raises(ValidationError):
        WebFetchParams(url="https://example.com", timeout=0)


def test_web_fetch_metadata_includes_matcher_and_preview() -> None:
    from aura.tools.web_fetch import web_fetch

    meta = web_fetch.metadata or {}
    assert meta.get("rule_matcher") is not None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"url": "https://x.com"}) == "url: https://x.com"
