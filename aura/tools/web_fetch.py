"""web_fetch tool — fetch a URL and return text content (stdlib urllib)."""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata

_DEFAULT_TIMEOUT = 30
_MAX_BYTES = 1024 * 1024


class WebFetchParams(BaseModel):
    url: str = Field(description="HTTP(S) URL to fetch.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT, ge=1, le=120,
        description="Timeout in seconds (1-120).",
    )


def _reject_private_host(host: str) -> None:
    """SSRF 防御：拒绝解析到私网 / 回环 / link-local / multicast / metadata IP 的 host。"""
    try:
        # getaddrinfo 覆盖 IPv4 + IPv6；任一结果落在私网就拒。
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


class WebFetch(BaseTool):
    name: str = "web_fetch"
    description: str = (
        "Fetch an HTTP(S) URL via GET. Returns text content (UTF-8 with "
        "replacement). Max 1 MB, truncates if larger. No auth / cookies."
    )
    args_schema: type[BaseModel] = WebFetchParams
    # NOT ``is_read_only``. ``is_read_only`` in the permission gate means
    # "safe to auto-approve, no prompt". Web fetch reaches external servers
    # — the URL alone, the user's IP, and any query-string tokens leave the
    # machine. Under prompt injection, the LLM could exfil data to an
    # attacker-controlled host. Our SSRF guard (``_reject_private_host``)
    # blocks internal-network scanning but NOT exfil. So: prompt by default.
    # Users who want automation either pre-allow specific URLs in
    # ``.aura/settings.json`` (``web_fetch(https://docs.python.org/...)``) or
    # run with ``--bypass-permissions``.
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

    def _run(self, url: str, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
        return _fetch(url=url, timeout=timeout)


web_fetch: BaseTool = WebFetch()
