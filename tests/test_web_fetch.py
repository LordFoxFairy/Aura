"""Tests for aura.tools.web_fetch."""

from __future__ import annotations

import importlib
import sys
from email.message import Message
from http.client import HTTPMessage
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult

from aura.domain.tool import ToolError, ValidationResult
from aura.domain.tool_meta_access import meta_dict
from aura.tools.web_fetch import (
    FetchedPage,
    WebFetch,
    WebFetchParams,
    WebFetchSuccess,
    _fetch,
    _ValidatingRedirectHandler,
    make_web_fetch,
)

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


def test_web_fetch_sends_standard_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_all_hosts(monkeypatch)
    captured: dict[str, Request] = {}

    def _fake_urlopen(req: Request, timeout: object) -> _FakeResponse:
        captured["req"] = req
        return _FakeResponse(b"ok")

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    _fetch(url="https://example.com")
    assert captured["req"].get_header("User-agent") == "aura/0.1.0"


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

    meta = meta_dict(web_fetch)
    # Deliberately NOT is_read_only — auto-approving network reach would
    # let a prompt-injected LLM exfiltrate via URL. See the class comment
    # in aura/tools/web_fetch.py.
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is False
    assert meta.get("is_concurrency_safe") is True


def test_web_fetch_timeout_bounds() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WebFetchParams(url="https://example.com", prompt="x", timeout=121)
    with pytest.raises(ValidationError):
        WebFetchParams(url="https://example.com", prompt="x", timeout=0)


def test_web_fetch_metadata_includes_matcher_and_preview() -> None:
    from aura.tools.web_fetch import web_fetch

    meta = meta_dict(web_fetch)
    assert meta.get("rule_matcher") is not None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"url": "https://x.com"}) == "url: https://x.com"


def test_validate_input_rejects_non_http_scheme() -> None:
    from aura.tools.web_fetch import web_fetch

    result = web_fetch.validate_input({"url": "file:///etc/passwd", "prompt": "x"})
    assert isinstance(result, ValidationResult)
    assert result.invalid is True
    assert "http(s)" in result.reason


def test_validate_input_rejects_url_without_host() -> None:
    from aura.tools.web_fetch import web_fetch

    result = web_fetch.validate_input({"url": "https://", "prompt": "x"})
    assert result.invalid is True
    assert "no host" in result.reason


def test_validate_input_accepts_https_url() -> None:
    from aura.tools.web_fetch import web_fetch

    result = web_fetch.validate_input(
        {"url": "https://example.com/page", "prompt": "x"},
    )
    assert result.invalid is False
    assert result.reason == ""


# --- _fetch error branches -------------------------------------------------


class _FakeHTTPError(HTTPError):
    """Real HTTPError subclass with a body whose read() can be made to fail."""

    def __init__(self, code: int, body: bytes, *, can_read: bool = True) -> None:
        super().__init__("https://example.com", code, "msg", Message(), None)
        self._body = body
        self._can_read = can_read

    def read(self, amt: int | None = None) -> bytes:
        if not self._can_read:
            raise OSError("body stream corrupt")
        if amt is None:
            return self._body
        return self._body[:amt]


def test_web_fetch_http_error_returns_body_for_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4xx must still surface its body so the summary path can describe it."""
    _allow_all_hosts(monkeypatch)

    def _fake_urlopen(req: object, timeout: object) -> _FakeResponse:
        raise _FakeHTTPError(404, b"not found page")

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    out = _fetch(url="https://example.com")
    assert out["status"] == 404
    assert out["content"] == "not found page"
    assert out["truncated"] is False
    assert out["content_type"] == ""


def test_web_fetch_http_error_corrupt_body_falls_back_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5xx whose body stream raises on read must degrade to '' not crash."""
    _allow_all_hosts(monkeypatch)

    def _fake_urlopen(req: object, timeout: object) -> _FakeResponse:
        raise _FakeHTTPError(503, b"ignored", can_read=False)

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    out = _fetch(url="https://example.com")
    assert out["status"] == 503
    assert out["content"] == ""


def test_web_fetch_timeout_error_raises_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket timeout must map to a ToolError naming the configured budget."""
    _allow_all_hosts(monkeypatch)

    def _fake_urlopen(req: object, timeout: object) -> _FakeResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(_wf_mod, "urlopen", _fake_urlopen)
    with pytest.raises(ToolError, match="timed out after 7s"):
        _fetch(url="https://example.com", timeout=7)


# --- _truncate_for_summary boundary matrix ---------------------------------


@pytest.mark.parametrize(
    ("size", "want_truncated", "want_len"),
    [
        (0, False, 0),
        (1, False, 1),
        (_wf_mod._SUMMARY_CONTENT_INPUT_CAP_CHARS, False, _wf_mod._SUMMARY_CONTENT_INPUT_CAP_CHARS),
        (
            _wf_mod._SUMMARY_CONTENT_INPUT_CAP_CHARS + 1,
            True,
            _wf_mod._SUMMARY_CONTENT_INPUT_CAP_CHARS,
        ),
    ],
)
def test_truncate_for_summary_caps_at_boundary(
    size: int, want_truncated: bool, want_len: int
) -> None:
    """Summary input must be capped exactly at the boundary, flag set only past it."""
    trimmed, truncated = _wf_mod._truncate_for_summary("a" * size)
    assert truncated is want_truncated
    assert len(trimmed) == want_len


# --- _model_name provider-attr matrix --------------------------------------


class _ModelNamed:
    def __init__(self, value: str) -> None:
        self.model_name = value


class _ModelAttr:
    def __init__(self, value: str) -> None:
        self.model = value


class _ModelBoth:
    """model_name wins over model when both are present + truthy."""

    def __init__(self, named: str, attr: str) -> None:
        self.model_name = named
        self.model = attr


class _ModelNone:
    pass


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        (_ModelNamed("gpt-4o-mini"), "gpt-4o-mini"),
        (_ModelAttr("claude-haiku"), "claude-haiku"),
        (_ModelBoth("primary", "secondary"), "primary"),
        (_ModelNamed(""), "_ModelNamed"),
        (_ModelNone(), "_ModelNone"),
    ],
)
def test_model_name_resolves_provider_attr(obj: object, expected: str) -> None:
    """Model id lives under model_name OR model; empty/absent falls to the class name."""
    assert _wf_mod._model_name(obj) == expected


# --- _Cache hit / expiry / eviction / LRU ----------------------------------


def _success(url: str = "https://example.com", summary: str = "s") -> WebFetchSuccess:
    return {
        "url": url,
        "status": 200,
        "summary": summary,
        "original_size_bytes": 1,
        "truncated_for_summary": False,
        "summary_model_name": "m",
        "cached": False,
    }


def test_cache_miss_returns_none() -> None:
    """An unseen key is a clean miss, never a stale partial."""
    cache = _wf_mod._Cache(max_entries=4, ttl=100.0)
    assert cache.get("absent") is None


def test_cache_hit_returns_stored_payload() -> None:
    """A fresh put must be retrievable byte-for-byte before TTL elapses."""
    cache = _wf_mod._Cache(max_entries=4, ttl=100.0)
    payload = _success(summary="cached-body")
    cache.put("k", payload)
    got = cache.get("k")
    assert got is not None
    assert got["summary"] == "cached-body"


def test_cache_expiry_evicts_stale_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past-TTL entries must be dropped on read so stale pages never resurface."""
    clock = {"now": 1_000.0}
    monkeypatch.setattr(_wf_mod.time, "time", lambda: clock["now"])
    cache = _wf_mod._Cache(max_entries=4, ttl=10.0)
    cache.put("k", _success())
    clock["now"] += 10.5  # just past ttl
    assert cache.get("k") is None
    # second read confirms the expired entry was physically deleted, not just hidden
    assert cache.get("k") is None


def test_cache_within_ttl_is_a_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry read exactly at the TTL edge (not strictly past) stays a hit."""
    clock = {"now": 500.0}
    monkeypatch.setattr(_wf_mod.time, "time", lambda: clock["now"])
    cache = _wf_mod._Cache(max_entries=4, ttl=10.0)
    cache.put("k", _success())
    clock["now"] += 10.0  # exactly ttl: diff == ttl is NOT > ttl
    assert cache.get("k") is not None


def test_cache_evicts_oldest_when_over_capacity() -> None:
    """Capacity overflow must drop the least-recently-used key, bounding memory."""
    cache = _wf_mod._Cache(max_entries=2, ttl=100.0)
    cache.put("a", _success(url="a"))
    cache.put("b", _success(url="b"))
    cache.put("c", _success(url="c"))  # forces eviction of "a"
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_cache_get_refreshes_lru_order() -> None:
    """Reading a key marks it recently-used so the next eviction spares it."""
    cache = _wf_mod._Cache(max_entries=2, ttl=100.0)
    cache.put("a", _success(url="a"))
    cache.put("b", _success(url="b"))
    cache.get("a")  # promote "a" ahead of "b"
    cache.put("c", _success(url="c"))  # now "b" is the LRU victim
    assert cache.get("b") is None
    assert cache.get("a") is not None


def test_cache_put_is_idempotent_on_same_key() -> None:
    """Re-putting a key overwrites in place — no duplicate entry, latest value wins."""
    cache = _wf_mod._Cache(max_entries=2, ttl=100.0)
    cache.put("k", _success(summary="v1"))
    cache.put("k", _success(summary="v2"))
    got = cache.get("k")
    assert got is not None
    assert got["summary"] == "v2"


# --- fake summary model + _run_summary -------------------------------------


class _FakeChat(BaseChatModel):
    """Minimal concrete BaseChatModel that echoes a canned str reply, no network."""

    reply: str = "digest"
    model_name: str = "fake-mini"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        raise NotImplementedError("async-only fake")

    def _make_message(self) -> AIMessage:
        return AIMessage(content=self.reply)

    async def ainvoke(
        self,
        input: object,
        config: object = None,
        *,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> AIMessage:
        return self._make_message()


class _ListContentChat(_FakeChat):
    """Returns multimodal list content, to exercise the str-coercion branch."""

    parts: list[str] = ["a", "b"]

    def _make_message(self) -> AIMessage:
        blocks: list[str | dict[str, str]] = list(self.parts)
        return AIMessage(content=blocks)


class _BoomChat(_FakeChat):
    """A model whose ainvoke explodes, to exercise the summary failure branch."""

    async def ainvoke(
        self,
        input: object,
        config: object = None,
        *,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> AIMessage:
        raise RuntimeError("upstream model 500")


async def test_run_summary_strips_str_content() -> None:
    """A str digest is returned trimmed, with truncation flag and resolved model id."""
    text, truncated, name = await _wf_mod._run_summary(
        model=_FakeChat(reply="  hi  "), prompt="q", body="body"
    )
    assert text == "hi"
    assert truncated is False
    assert name == "fake-mini"


async def test_run_summary_coerces_non_str_content() -> None:
    """Multimodal/list content must be coerced to str, never leak a raw list out."""
    text, _truncated, _name = await _wf_mod._run_summary(
        model=_ListContentChat(), prompt="q", body="body"
    )
    assert isinstance(text, str)
    assert "a" in text


async def test_run_summary_flags_oversized_body() -> None:
    """A body past the input cap reports truncated=True so callers know it was clipped."""
    big = "z" * (_wf_mod._SUMMARY_CONTENT_INPUT_CAP_CHARS + 10)
    _text, truncated, _name = await _wf_mod._run_summary(
        model=_FakeChat(), prompt="q", body=big
    )
    assert truncated is True


# --- _resolve_factory ------------------------------------------------------


def test_resolve_factory_prefers_instance_factory() -> None:
    """A per-instance factory overrides the module default (DI beats global wiring)."""
    sentinel = _FakeChat()
    tool = make_web_fetch(lambda: sentinel)
    assert tool._resolve_factory()() is sentinel


def test_resolve_factory_falls_back_to_module_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no instance factory, the module-level default is used."""
    sentinel = _FakeChat()
    monkeypatch.setattr(_wf_mod, "_DEFAULT_MODEL_FACTORY", lambda: sentinel)
    tool = WebFetch()
    assert tool._resolve_factory()() is sentinel


def test_resolve_factory_unconfigured_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No factory anywhere must fail loud, not silently summarise with nothing."""
    monkeypatch.setattr(_wf_mod, "_DEFAULT_MODEL_FACTORY", None)
    tool = WebFetch()
    with pytest.raises(ToolError, match="no summary model factory"):
        tool._resolve_factory()


# --- WebFetch._arun end-to-end (cache, bypass, failure payloads) ------------


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, body: str, status: int = 200) -> None:
    def _fake(*, url: str, timeout: int) -> FetchedPage:
        return {
            "url": url,
            "status": status,
            "content_type": "text/plain",
            "content": body,
            "truncated": False,
        }

    monkeypatch.setattr(_wf_mod, "_fetch", _fake)


def _fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_wf_mod, "_CACHE", _wf_mod._Cache())


def test_arun_sync_run_is_disabled() -> None:
    """The sync path must refuse — web_fetch is async-only and must say so loudly."""
    with pytest.raises(NotImplementedError, match="async-only"):
        WebFetch(model_factory=lambda: _FakeChat())._run(url="https://x", prompt="q")


async def test_arun_success_summarises_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean fetch yields a summary success, records size, and warms the cache."""
    _fresh_cache(monkeypatch)
    _patch_fetch(monkeypatch, body="page body")
    tool = make_web_fetch(lambda: _FakeChat(reply="the digest"))
    out = await tool._arun(url="https://example.com", prompt="q")
    assert out["summary"] is not None
    assert out["summary"] == "the digest"
    assert out["status"] == 200
    assert out["original_size_bytes"] == len(b"page body")
    assert out["cached"] is False
    assert out["summary_model_name"] == "fake-mini"


async def test_arun_second_call_is_a_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotency: a repeat url+prompt must serve from cache (cached flag flips True)."""
    _fresh_cache(monkeypatch)
    calls = {"n": 0}

    def _counting_factory() -> _FakeChat:
        calls["n"] += 1
        return _FakeChat(reply="once")

    _patch_fetch(monkeypatch, body="b")
    tool = make_web_fetch(_counting_factory)
    first = await tool._arun(url="https://example.com", prompt="q")
    second = await tool._arun(url="https://example.com", prompt="q")
    assert first["summary"] is not None
    assert second["summary"] is not None
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["summary"] == "once"
    assert calls["n"] == 1  # model invoked only on the miss


async def test_arun_bypass_cache_refetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """bypass_cache=True must skip a warm entry and re-run the model every time."""
    _fresh_cache(monkeypatch)
    calls = {"n": 0}

    def _counting_factory() -> _FakeChat:
        calls["n"] += 1
        return _FakeChat(reply=f"call-{calls['n']}")

    _patch_fetch(monkeypatch, body="b")
    tool = make_web_fetch(_counting_factory)
    await tool._arun(url="https://example.com", prompt="q")
    out = await tool._arun(url="https://example.com", prompt="q", bypass_cache=True)
    assert out["summary"] is not None
    assert out["cached"] is False
    assert calls["n"] == 2


async def test_arun_factory_failure_returns_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model-factory crash must degrade to a structured failure, never propagate."""
    _fresh_cache(monkeypatch)
    _patch_fetch(monkeypatch, body="raw page", status=200)

    def _bad_factory() -> BaseChatModel:
        raise RuntimeError("no api key")

    tool = make_web_fetch(_bad_factory)
    out = await tool._arun(url="https://example.com", prompt="q")
    assert out["summary"] is None
    assert "summary factory failed" in out["error"]
    assert out["raw_body_preview"] == "raw page"
    assert out["summary_model_name"] is None
    assert out["status"] == 200


async def test_arun_invoke_failure_returns_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model-invoke crash must degrade to a structured failure carrying the body."""
    _fresh_cache(monkeypatch)
    _patch_fetch(monkeypatch, body="raw page", status=502)
    tool = make_web_fetch(lambda: _BoomChat())
    out = await tool._arun(url="https://example.com", prompt="q")
    assert out["summary"] is None
    assert "summary invoke failed" in out["error"]
    assert out["status"] == 502


async def test_arun_failure_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure must NOT stick in cache — the retry must hit the model again."""
    _fresh_cache(monkeypatch)
    calls = {"n": 0}

    def _counting_boom() -> _BoomChat:
        calls["n"] += 1
        return _BoomChat()

    _patch_fetch(monkeypatch, body="b")
    tool = make_web_fetch(_counting_boom)
    first = await tool._arun(url="https://example.com", prompt="q")
    second = await tool._arun(url="https://example.com", prompt="q")
    assert first["summary"] is None
    assert second["summary"] is None  # still a failure, never served stale-cached
    assert calls["n"] == 2  # model rebuilt on retry → first failure was not cached


def test_redirect_to_cloud_metadata_ip_is_rejected() -> None:
    """SSRF: a 302 to the 169.254.169.254 metadata endpoint must be refused — the
    original-host guard alone would let urlopen follow the redirect and exfiltrate creds."""
    handler = _ValidatingRedirectHandler()
    req = Request("https://example.com/")
    with pytest.raises(ToolError, match="non-public IP"):
        handler.redirect_request(
            req, BytesIO(b""), 302, "Found", HTTPMessage(),
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        )


def test_redirect_to_loopback_is_rejected() -> None:
    """A redirect to 127.0.0.1 (internal service) must be refused on the hop."""
    handler = _ValidatingRedirectHandler()
    with pytest.raises(ToolError, match="non-public IP"):
        handler.redirect_request(
            Request("https://example.com/"), BytesIO(b""), 302, "Found",
            HTTPMessage(), "http://127.0.0.1:8080/admin",
        )


def test_redirect_to_non_http_scheme_is_rejected() -> None:
    """A redirect to file:// (or any non-http(s) scheme) must be refused before fetch."""
    handler = _ValidatingRedirectHandler()
    with pytest.raises(ToolError, match="non-http"):
        handler.redirect_request(
            Request("https://example.com/"), BytesIO(b""), 302, "Found",
            HTTPMessage(), "file:///etc/passwd",
        )
