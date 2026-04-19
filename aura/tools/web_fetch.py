"""web_fetch tool — fetch a URL and return text content (stdlib urllib)."""

from __future__ import annotations

from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from aura.tools.base import AuraTool, ToolResult, build_tool  # noqa: TC001

_DEFAULT_TIMEOUT = 30
_MAX_BYTES = 1024 * 1024


class WebFetchParams(BaseModel):
    url: str = Field(description="HTTP(S) URL to fetch.")
    timeout: int = Field(
        default=_DEFAULT_TIMEOUT, ge=1, le=120,
        description="Timeout in seconds (1-120).",
    )


def _fetch(params: WebFetchParams) -> ToolResult:
    if not (params.url.startswith("http://") or params.url.startswith("https://")):
        return ToolResult(ok=False, error=f"not an http(s) URL: {params.url}")

    req = Request(params.url, headers={"User-Agent": "aura/0.1.0"})
    try:
        with urlopen(req, timeout=params.timeout) as resp:  # noqa: S310
            data = resp.read(_MAX_BYTES + 1)
            status = resp.status
            content_type = resp.headers.get("Content-Type", "") or ""
    except URLError as exc:
        return ToolResult(ok=False, error=f"fetch failed: {exc}")
    except TimeoutError as exc:
        return ToolResult(ok=False, error=f"fetch timed out after {params.timeout}s: {exc}")

    truncated = len(data) > _MAX_BYTES
    if truncated:
        data = data[:_MAX_BYTES]
    content = data.decode("utf-8", errors="replace")

    output: dict[str, Any] = {
        "url": params.url,
        "status": status,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
    }
    return ToolResult(ok=True, output=output)


web_fetch: AuraTool = build_tool(
    name="web_fetch",
    description=(
        "Fetch an HTTP(S) URL via GET. Returns text content (UTF-8 with "
        "replacement). Max 1 MB, truncates if larger. No auth / cookies."
    ),
    input_model=WebFetchParams,
    call=_fetch,
    is_read_only=True,
    is_concurrency_safe=True,
    max_result_size_chars=60_000,
)
