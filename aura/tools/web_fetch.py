"""web_fetch — GET a URL, summarise via cheap model, optional cache."""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import socket
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol, TypedDict, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.domain.permission.matchers import exact_match_on
from aura.domain.tool import ToolError, ToolMetadata, ValidationResult
from aura.tools.base import Tool

_DEFAULT_TIMEOUT = 30
_MAX_BYTES = 1024 * 1024
_PROMPT_MAX_CHARS = 4_000

_CACHE_TTL_SEC = 15 * 60
_CACHE_MAX_ENTRIES = 64

# Module-level since every AgentSession shares the same web_fetch singleton.
_DEFAULT_MODEL_FACTORY: Callable[[], BaseChatModel] | None = None


def set_default_model_factory(
    factory: Callable[[], BaseChatModel] | None,
) -> None:
    global _DEFAULT_MODEL_FACTORY
    _DEFAULT_MODEL_FACTORY = factory


class WebFetchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="HTTP(S) URL to fetch.")
    prompt: str = Field(
        ..., min_length=1, max_length=_PROMPT_MAX_CHARS,
        description="Question / extraction goal for summarising the page.",
    )
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT, ge=1, le=120,
        description="Timeout in seconds (1-120).",
    )
    bypass_cache: bool = Field(
        default=False,
        description="When True, skip the 15-min cache and re-fetch.",
    )


def _reject_private_host(host: str) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ToolError(f"dns resolve failed for {host!r}: {exc}") from exc

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ToolError(
                f"refusing to fetch {host!r} — resolves to non-public IP {addr}"
            )


class FetchedPage(TypedDict):
    url: str
    status: int
    content_type: str
    content: str
    truncated: bool


class WebFetchSuccess(TypedDict):
    url: str
    status: int | None
    summary: str
    original_size_bytes: int
    truncated_for_summary: bool
    summary_model_name: str
    cached: bool


class WebFetchFailure(TypedDict):
    url: str
    status: int | None
    summary: None
    error: str
    raw_body_preview: str
    truncated_for_summary: bool
    summary_model_name: None


WebFetchResult = WebFetchSuccess | WebFetchFailure


def _fetch(url: str, timeout: int = _DEFAULT_TIMEOUT) -> FetchedPage:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError(f"not an http(s) URL: {url}")

    parsed = urlparse(url)
    if not parsed.hostname:
        raise ToolError(f"malformed URL (no host): {url}")
    _reject_private_host(parsed.hostname)

    req = Request(url, headers={"User-AgentSession": "aura/0.1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — scheme + private-host already validated above
            data = resp.read(_MAX_BYTES + 1)
            status = resp.status
            content_type = resp.headers.get("Content-Type", "") or ""
    except HTTPError as exc:
        # HTTPError is response-like; read body so summary path can describe it.
        try:
            body = exc.read(_MAX_BYTES + 1) if hasattr(exc, "read") else b""
        except Exception:  # noqa: BLE001  # corrupt input falls back to default
            body = b""
        return {
            "url": url,
            "status": exc.code,
            "content_type": "",
            "content": body.decode("utf-8", errors="replace"),
            "truncated": False,
        }
    except URLError as exc:
        raise ToolError(f"fetch failed: {exc}") from exc
    except TimeoutError as exc:
        raise ToolError(f"fetch timed out after {timeout}s: {exc}") from exc

    truncated = len(data) > _MAX_BYTES
    if truncated:
        data = data[:_MAX_BYTES]
    content = data.decode("utf-8", errors="replace")

    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
    }


def _preview(args: dict[str, Any]) -> str:
    return f"url: {args.get('url', '')}"


def _cache_key(url: str, prompt: str) -> str:
    return f"{url}\x00{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"


class _Cache:
    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES, ttl: float = _CACHE_TTL_SEC) -> None:
        self._entries: OrderedDict[str, tuple[float, WebFetchSuccess]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl

    def get(self, key: str) -> WebFetchSuccess | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if time.time() - ts > self._ttl:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return payload

    def put(self, key: str, payload: WebFetchSuccess) -> None:
        self._entries[key] = (time.time(), payload)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def clear(self) -> None:  # pragma: no cover — used only in tests
        self._entries.clear()


_CACHE = _Cache()


_SUMMARY_CONTENT_INPUT_CAP_CHARS = 60_000


def _truncate_for_summary(content: str) -> tuple[str, bool]:
    if len(content) <= _SUMMARY_CONTENT_INPUT_CAP_CHARS:
        return content, False
    return content[:_SUMMARY_CONTENT_INPUT_CAP_CHARS], True


def _build_summary_prompt(prompt: str, body: str) -> str:
    return (
        "You are summarising a fetched web page for an AI agent. The "
        "agent provided a focused question; reply with a concise digest "
        "(no preamble, no boilerplate) that answers it from the page "
        "body below.\n\n"
        f"Question: {prompt}\n\n"
        "Page body:\n"
        f"{body}"
    )


# Provider subclasses expose the model id under varying attrs:
# Anthropic/others use model_name; OpenAI uses model.
@runtime_checkable
class _HasModelName(Protocol):
    model_name: str


@runtime_checkable
class _HasModel(Protocol):
    model: str


def _model_name(model: BaseChatModel) -> str:
    if isinstance(model, _HasModelName) and model.model_name:
        name: object = model.model_name
    elif isinstance(model, _HasModel) and model.model:
        name = model.model
    else:
        name = None
    return name if isinstance(name, str) else type(model).__name__


async def _run_summary(
    *,
    model: BaseChatModel,
    prompt: str,
    body: str,
) -> tuple[str, bool, str]:
    trimmed, truncated = _truncate_for_summary(body)
    digest_prompt = _build_summary_prompt(prompt, trimmed)
    ai = await model.ainvoke([HumanMessage(content=digest_prompt)])
    text = ai.content if isinstance(ai.content, str) else str(ai.content)
    return text.strip(), truncated, _model_name(model)


_RAW_BODY_PREVIEW_CHARS = 500


def _failure_payload(
    url: str,
    status: int | None,
    error: str,
    content: str,
    truncated_for_summary: bool,
) -> WebFetchFailure:
    return {
        "url": url,
        "status": status,
        "summary": None,
        "error": error,
        "raw_body_preview": content[:_RAW_BODY_PREVIEW_CHARS],
        "truncated_for_summary": truncated_for_summary,
        "summary_model_name": None,
    }


class WebFetch(Tool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "web_fetch"
    description: str = (
        "Fetch an HTTP(S) URL via GET, then summarise the body using a "
        "cheap model + the caller's `prompt`. Max 1 MB body, truncates "
        "if larger. SSRF-guarded. 15-min in-process cache (keyed by "
        "url+prompt). No auth / cookies."
    )
    args_schema: type[BaseModel] = WebFetchParams  # pyright: ignore[reportIncompatibleVariableOverride]  # langchain BaseTool declares args_schema as mutable ArgsSchema|None; subclass narrows widely on purpose.
    # NOT is_read_only — fetching exfils request data over the network.
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=False,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=exact_match_on("url"),
        args_preview=_preview,
        timeout_sec=30.0,
    )

    _instance_factory: Callable[[], BaseChatModel] | None = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        model_factory: Callable[[], BaseChatModel] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._instance_factory = model_factory

    def _resolve_factory(self) -> Callable[[], BaseChatModel]:
        if self._instance_factory is not None:
            return self._instance_factory
        if _DEFAULT_MODEL_FACTORY is not None:
            return _DEFAULT_MODEL_FACTORY
        raise ToolError(
            "web_fetch: no summary model factory configured — "
            "AgentSession.__init__ usually wires this; build the tool with "
            "make_web_fetch(factory) for SDK use.",
        )

    def validate_input(self, args: dict[str, Any]) -> ValidationResult:
        url = args.get("url", "")
        if not isinstance(url, str) or not (
            url.startswith("http://") or url.startswith("https://")
        ):
            return ValidationResult(
                invalid=True,
                reason=f"not an http(s) URL: {url}",
            )
        parsed = urlparse(url)
        if not parsed.hostname:
            return ValidationResult(
                invalid=True,
                reason=f"malformed URL (no host): {url}",
            )
        return ValidationResult(invalid=False)

    def _run(
        self,
        url: str,
        prompt: str,
        timeout: int = _DEFAULT_TIMEOUT,
        bypass_cache: bool = False,
    ) -> WebFetchResult:
        raise NotImplementedError("web_fetch is async-only; use ainvoke")

    async def _arun(
        self,
        url: str,
        prompt: str,
        timeout: int = _DEFAULT_TIMEOUT,
        bypass_cache: bool = False,
    ) -> WebFetchResult:
        key = _cache_key(url, prompt)
        if not bypass_cache:
            cached = _CACHE.get(key)
            if cached is not None:
                hit: WebFetchSuccess = {**cached, "cached": True}
                return hit

        fetched = _fetch(url=url, timeout=timeout)
        body = fetched["content"]
        factory = self._resolve_factory()
        try:
            summary_model = factory()
        except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            return _failure_payload(
                url=url,
                status=fetched["status"],
                error=f"summary factory failed: {type(exc).__name__}: {exc}",
                content=body,
                truncated_for_summary=False,
            )
        try:
            summary_text, truncated, model_name = await _run_summary(
                model=summary_model, prompt=prompt, body=body,
            )
        except Exception as exc:  # noqa: BLE001  # swallowed at boundary; failure must not propagate
            return _failure_payload(
                url=url,
                status=fetched["status"],
                error=f"summary invoke failed: {type(exc).__name__}: {exc}",
                content=body,
                truncated_for_summary=False,
            )

        result: WebFetchSuccess = {
            "url": url,
            "status": fetched["status"],
            "summary": summary_text,
            "original_size_bytes": len(body.encode("utf-8")),
            "truncated_for_summary": truncated,
            "summary_model_name": model_name,
            "cached": False,
        }
        # Cache successes only — keeps a transient 4xx from sticking for 15 min.
        with contextlib.suppress(Exception):
            _CACHE.put(key, result)
        return result


def make_web_fetch(
    model_factory: Callable[[], BaseChatModel] | None = None,
) -> WebFetch:
    return WebFetch(model_factory=model_factory)


web_fetch: WebFetch = WebFetch()
