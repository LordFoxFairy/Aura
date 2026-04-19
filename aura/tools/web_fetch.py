"""web_fetch tool — fetch a URL and return text content (stdlib urllib)."""

from __future__ import annotations

from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.tools.base import ToolError, build_tool

_DEFAULT_TIMEOUT = 30
_MAX_BYTES = 1024 * 1024


class WebFetchParams(BaseModel):
    url: str = Field(description="HTTP(S) URL to fetch.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT, ge=1, le=120,
        description="Timeout in seconds (1-120).",
    )


def _fetch(url: str, timeout: int = _DEFAULT_TIMEOUT) -> dict[str, Any]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError(f"not an http(s) URL: {url}")

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


web_fetch: BaseTool = build_tool(
    name="web_fetch",
    description=(
        "Fetch an HTTP(S) URL via GET. Returns text content (UTF-8 with "
        "replacement). Max 1 MB, truncates if larger. No auth / cookies."
    ),
    args_schema=WebFetchParams,
    func=_fetch,
    is_read_only=True,
    is_concurrency_safe=True,
    max_result_size_chars=60_000,
)
