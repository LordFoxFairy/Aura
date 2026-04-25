"""web_fetch tool — fetch a URL, summarise via cheap model, optional cache.

Three concerns layered onto the basic GET:

- **Round 1C SSRF defense**. ``_reject_private_host`` rejects DNS that
  resolves to private / loopback / link-local / multicast / metadata
  IPs (including the AWS / GCP / Azure ``169.254.169.254`` instance
  metadata endpoint). The :class:`_RedirectGuard` re-runs the same
  check on every redirect so a 302 to ``http://localhost`` doesn't
  bypass the gate. Capped at 5 redirects.
- **Round 4E summary pipeline**. The fetched body is fed (with the
  caller's prompt) to a cheap summary model — the result is a tight
  digest the LLM can act on instead of the raw HTML. The model
  factory is set globally by Agent.__init__ via
  :func:`set_default_model_factory`; tests can also build a per-instance
  tool via :func:`make_web_fetch`.
- **Round 6N cache**. 15-minute TTL, LRU at 64 entries. Keyed by
  ``(url, sha256(prompt))`` so the same URL with different prompts
  don't share a slot. ``bypass_cache=True`` forces a refetch.

Failure shape: every error path returns a dict (not a raise) with
``status`` set to whatever the upstream said (or ``None`` for
pre-flight rejection), ``summary=None``, ``error`` populated, and
``raw_body_preview`` carrying the first ~500 chars of the body that
DID arrive (helps the LLM diagnose 4xx/5xx without re-fetching).
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import socket
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    urlopen,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata

_DEFAULT_TIMEOUT = 30
_MAX_BYTES = 1024 * 1024
_MAX_REDIRECTS = 5
_PROMPT_MAX_CHARS = 4_000

# Round 6N cache. 15-minute TTL is long enough to amortise cost across a
# multi-turn conversation that revisits the same URL ("now look at the
# class hierarchy"), short enough that a doc page that updated this hour
# isn't permanently stale. 64 entries keeps the working-set small —
# this is a per-process cache, not a CDN.
_CACHE_TTL_SEC = 15 * 60
_CACHE_MAX_ENTRIES = 64

# Module-level model factory, wired by Agent.__init__ via
# :func:`set_default_model_factory`. ``None`` means "no factory wired" —
# the singleton then raises a clear ToolError on first invoke (rather
# than silently swallowing the lack of a summary).
_DEFAULT_MODEL_FACTORY: Callable[[], BaseChatModel] | None = None


def set_default_model_factory(
    factory: Callable[[], BaseChatModel] | None,
) -> None:
    """Wire the module-level summary model factory.

    Called by ``Agent.__init__`` once at construction time. Passing
    ``None`` clears the factory (used between tests that build multiple
    Agents in series).

    Why a module-level global rather than per-instance: there's exactly
    one shared ``web_fetch`` singleton in ``BUILTIN_TOOLS`` so every
    Agent shares it. Wiring the factory at the singleton would mean
    the LAST-built Agent's summary model is what every Agent uses —
    deliberate, since Agents per-process are normally one in number.
    """
    global _DEFAULT_MODEL_FACTORY
    _DEFAULT_MODEL_FACTORY = factory


class _RedirectGuard(HTTPRedirectHandler):
    """Re-run SSRF check on every redirect; cap at :data:`_MAX_REDIRECTS`.

    Stock urllib.HTTPRedirectHandler doesn't expose a per-redirect hook,
    but ``redirect_request`` is called for every 30x — we override and
    re-validate the new host before returning a Request, so a 302 to
    ``http://localhost`` from an attacker-controlled site can't pierce
    the gate.
    """

    max_redirections = _MAX_REDIRECTS

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        # Re-check on every hop. ``_reject_private_host`` raises
        # ToolError; let it propagate — urllib catches Exception in
        # the outer urlopen and we surface it as a fetch_failed dict.
        new_host = urlparse(newurl).hostname
        if new_host is not None:
            _reject_private_host(new_host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class WebFetchParams(BaseModel):
    """``WebFetch`` input schema — Round 4E summary pipeline.

    ``prompt`` is REQUIRED — without it the tool can't produce a
    summary. Capped at 4_000 chars so a misbehaving LLM doesn't try
    to ship the entire conversation history through here.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = Field(description="HTTP(S) URL to fetch.")
    prompt: str = Field(
        ..., min_length=1, max_length=_PROMPT_MAX_CHARS,
        description=(
            "Question / extraction goal the cheap model uses to "
            "summarise the fetched page. Keep concise — capped at 4000 chars."
        ),
    )
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT, ge=1, le=120,
        description="Timeout in seconds (1-120).",
    )
    bypass_cache: bool = Field(
        default=False,
        description=(
            "When True, skip the 15-min cache lookup and re-fetch. "
            "Use sparingly — the cache pays off across multi-turn "
            "conversations that revisit the same URL."
        ),
    )


def _reject_private_host(host: str) -> None:
    """SSRF defense: refuse hosts that resolve to non-public IPs."""
    try:
        # getaddrinfo covers IPv4 + IPv6; any private result rejects.
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


def _fetch(url: str, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Lower-level fetcher returning the raw response shape.

    Kept as a module-level function (not a method) so the test suite can
    monkeypatch ``aura.tools.web_fetch._fetch`` to short-circuit the
    network in tests of the wider summary pipeline.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError(f"not an http(s) URL: {url}")

    parsed = urlparse(url)
    if not parsed.hostname:
        raise ToolError(f"malformed URL (no host): {url}")
    _reject_private_host(parsed.hostname)

    req = Request(url, headers={"User-Agent": "aura/0.1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read(_MAX_BYTES + 1)
            status = resp.status
            content_type = resp.headers.get("Content-Type", "") or ""
    except HTTPError as exc:  # 4xx / 5xx — surface status + body preview
        # ``HTTPError`` IS a Response-like object — read the partial body
        # so the summary path can still produce a useful diagnosis.
        try:
            body = exc.read(_MAX_BYTES + 1) if hasattr(exc, "read") else b""
        except Exception:  # noqa: BLE001 — body read can fail on closed stream
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

    output: dict[str, Any] = {
        "url": url,
        "status": status,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
    }
    return output


def _preview(args: dict[str, Any]) -> str:
    return f"url: {args.get('url', '')}"


def _cache_key(url: str, prompt: str) -> str:
    """Derive a cache key from URL + prompt hash.

    Hashing the prompt (rather than embedding it) keeps the key short
    + avoids leaking the prompt text into log lines that include the
    cache key for diagnostics.
    """
    return f"{url}\x00{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"


class _Cache:
    """Tiny LRU + TTL cache for summary results.

    OrderedDict + manual move-to-end on get is the textbook minimal
    LRU. TTL is checked on read so an entry can age out without a
    background sweep.
    """

    def __init__(self, max_entries: int = _CACHE_MAX_ENTRIES, ttl: float = _CACHE_TTL_SEC) -> None:
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if time.time() - ts > self._ttl:
            # Expired — drop it so the next miss doesn't keep tripping
            # over a stale entry.
            del self._entries[key]
            return None
        # LRU bump on access.
        self._entries.move_to_end(key)
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._entries[key] = (time.time(), payload)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max:
            self._entries.popitem(last=False)

    def clear(self) -> None:  # pragma: no cover — used only in tests
        self._entries.clear()


_CACHE = _Cache()


# Limits applied to the body BEFORE it's fed to the summary model.
# The model factory might be a frontier model with a 200k context, but
# burning 200k input tokens to summarise one HTML page is wasteful; cap
# the input so the cheap model stays cheap.
_SUMMARY_CONTENT_INPUT_CAP_CHARS = 60_000


def _truncate_for_summary(content: str) -> tuple[str, bool]:
    """Trim ``content`` to :data:`_SUMMARY_CONTENT_INPUT_CAP_CHARS`.

    Returns ``(trimmed, was_truncated)``. We keep the head (most pages
    front-load the meaningful content); a future improvement could
    walk to the first ``<body>`` tag, but the simple head-keep
    catches the common case.
    """
    if len(content) <= _SUMMARY_CONTENT_INPUT_CAP_CHARS:
        return content, False
    return content[:_SUMMARY_CONTENT_INPUT_CAP_CHARS], True


def _build_summary_prompt(prompt: str, body: str) -> str:
    """Render the cheap-model prompt envelope.

    Keeps the structure tight + delimited so the summary model has a
    clear "follow this question, look in this body" frame.
    """
    return (
        "You are summarising a fetched web page for an AI agent. The "
        "agent provided a focused question; reply with a concise digest "
        "(no preamble, no boilerplate) that answers it from the page "
        "body below.\n\n"
        f"Question: {prompt}\n\n"
        "Page body:\n"
        f"{body}"
    )


def _model_name(model: BaseChatModel) -> str:
    """Best-effort: extract a human-readable name for the summary model.

    LangChain's chat models expose either ``model_name`` or ``model``
    (different SDK conventions). We try both, fall back to the class
    name. Used purely for the result payload — never for routing.
    """
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


async def _run_summary(
    *,
    model: BaseChatModel,
    prompt: str,
    body: str,
) -> tuple[str, bool, str]:
    """Invoke the cheap model. Returns ``(text, truncated, model_name)``."""
    trimmed, truncated = _truncate_for_summary(body)
    digest_prompt = _build_summary_prompt(prompt, trimmed)
    ai = await model.ainvoke([HumanMessage(content=digest_prompt)])
    # ``content`` may be str (most providers) or list[dict] (Anthropic
    # multi-part). Normalise to str.
    text = ai.content if isinstance(ai.content, str) else str(ai.content)
    return text.strip(), truncated, _model_name(model)


_RAW_BODY_PREVIEW_CHARS = 500


def _failure_payload(
    url: str,
    status: int | None,
    error: str,
    content: str,
    truncated_for_summary: bool,
) -> dict[str, Any]:
    return {
        "url": url,
        "status": status,
        "summary": None,
        "error": error,
        "raw_body_preview": content[:_RAW_BODY_PREVIEW_CHARS],
        "truncated_for_summary": truncated_for_summary,
        "summary_model_name": None,
    }


class WebFetch(BaseTool):
    """Fetch + summarise via a cheap model with optional caching."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "web_fetch"
    description: str = (
        "Fetch an HTTP(S) URL via GET, then summarise the body using a "
        "cheap model + the caller's `prompt`. Max 1 MB body, truncates "
        "if larger. SSRF-guarded. 15-min in-process cache (keyed by "
        "url+prompt). No auth / cookies."
    )
    args_schema: type[BaseModel] = WebFetchParams
    # NOT ``is_read_only``. ``is_read_only`` in the permission gate means
    # "safe to auto-approve, no prompt". Web fetch reaches external servers
    # — the URL alone, the user's IP, and any query-string tokens leave the
    # machine. Under prompt injection, the LLM could exfil data to an
    # attacker-controlled host. Our SSRF guard (``_reject_private_host``)
    # blocks internal-network scanning but NOT exfil. So: prompt by default.
    metadata: dict[str, Any] | None = tool_metadata(
        is_concurrency_safe=True,
        max_result_size_chars=60_000,
        rule_matcher=exact_match_on("url"),
        args_preview=_preview,
        # 30s outer deadline. ``WebFetchParams.timeout`` still caps the
        # inner urlopen at its own value (default 30s) — the wait_for wrap
        # covers the socket-hang case where the internal timeout is
        # bypassed (DNS resolver stalls, TLS handshake stuck, etc.).
        timeout_sec=30.0,
        # Fetched documents are user-requested; do NOT fold.
        is_search_command=False,
    )

    # Optional per-instance factory (test override). When None, the
    # tool falls back to the module-level ``_DEFAULT_MODEL_FACTORY``
    # set by Agent.__init__.
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
            "Agent.__init__ usually wires this; build the tool with "
            "make_web_fetch(factory) for SDK use.",
        )

    def _run(
        self,
        url: str,
        prompt: str,
        timeout: int = _DEFAULT_TIMEOUT,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError("web_fetch is async-only; use ainvoke")

    async def _arun(
        self,
        url: str,
        prompt: str,
        timeout: int = _DEFAULT_TIMEOUT,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        # Cache check FIRST (before the SSRF/network round-trip). Hits
        # are cheap; misses fall through to the full pipeline.
        key = _cache_key(url, prompt)
        if not bypass_cache:
            cached = _CACHE.get(key)
            if cached is not None:
                # Annotate the cache hit so callers can tell a hot
                # answer from a fresh fetch — useful for "force a
                # re-fetch via bypass_cache" debugging.
                hit = dict(cached)
                hit["cached"] = True
                return hit

        # Network fetch. Body errors fall through to the summary path
        # so the model can describe a 404 / 500 if useful; transport
        # errors become ToolError raises (signal vs. body distinction).
        try:
            fetched = _fetch(url=url, timeout=timeout)
        except ToolError:
            # Reraise — the calling tool surface already turns ToolError
            # into the standard failure shape.
            raise

        # Empty body → still try summary. Some endpoints return 204 with
        # the meaningful info in headers; the model is allowed to
        # describe that.
        body = fetched.get("content", "")

        # Summary leg.
        try:
            factory = self._resolve_factory()
        except ToolError:
            raise
        try:
            summary_model = factory()
        except Exception as exc:  # noqa: BLE001
            return _failure_payload(
                url=url,
                status=fetched.get("status"),
                error=f"summary factory failed: {type(exc).__name__}: {exc}",
                content=body,
                truncated_for_summary=False,
            )
        try:
            summary_text, truncated, model_name = await _run_summary(
                model=summary_model, prompt=prompt, body=body,
            )
        except Exception as exc:  # noqa: BLE001
            return _failure_payload(
                url=url,
                status=fetched.get("status"),
                error=f"summary invoke failed: {type(exc).__name__}: {exc}",
                content=body,
                truncated_for_summary=False,
            )

        result = {
            "url": url,
            "status": fetched.get("status"),
            "summary": summary_text,
            "original_size_bytes": len(body.encode("utf-8")),
            "truncated_for_summary": truncated,
            "summary_model_name": model_name,
            "cached": False,
        }
        # Cache success only — failure shapes don't enter the cache so
        # a transient blip doesn't lock in a 4xx for 15 minutes.
        with contextlib.suppress(Exception):
            _CACHE.put(key, result)
        return result


def make_web_fetch(
    model_factory: Callable[[], BaseChatModel] | None = None,
) -> WebFetch:
    """Build a fresh :class:`WebFetch` with a per-instance factory.

    Used by tests + SDK callers that want to bypass the module-level
    singleton (e.g. building two Agents with different summary models in
    the same process).
    """
    return WebFetch(model_factory=model_factory)


# Module-level singleton — composed into ``BUILTIN_TOOLS`` and consumed by
# Agent.__init__. The first invocation without a wired factory raises a
# clear ToolError pointing at ``set_default_model_factory``; production
# callers always have it set before the LLM ever invokes the tool.
web_fetch: BaseTool = WebFetch()


# Bind the wiring helpers onto the singleton itself so call sites that
# resolve through the ``aura.tools.__init__`` re-export
# (``from aura.tools import web_fetch`` returns the SINGLETON, not the
# module) can still reach the wiring entry point via getattr. Both
# ``module.set_default_model_factory`` and
# ``singleton.set_default_model_factory`` end up calling the same
# module-level function — the static method is just a forwarding shim.
_set_default_factory_attr = staticmethod(set_default_model_factory)
_make_web_fetch_attr = staticmethod(make_web_fetch)
# ``object.__setattr__`` because pydantic's BaseTool blocks regular
# attribute assignment on instance for fields not declared on the
# model. The singleton-attached helpers aren't fields; they're
# convenience shortcuts to the module functions.
object.__setattr__(
    web_fetch, "set_default_model_factory", set_default_model_factory,
)
object.__setattr__(web_fetch, "make_web_fetch", make_web_fetch)
