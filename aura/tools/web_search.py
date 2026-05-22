"""web_search — DuckDuckGo query → list of {title, url, snippet}."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.config.schema import WebSearchConfig
from aura.domain.permission.matchers import exact_match_on
from aura.schemas.tool import ToolError, ToolMetadata

try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException

    _HAS_DDGS = True
except ImportError:  # pragma: no cover — exercised via monkeypatch in tests
    DDGS = None  # type: ignore[assignment,misc]
    RatelimitException = Exception  # type: ignore[assignment,misc]
    _HAS_DDGS = False


_INSTALL_HINT = (
    "web_search requires the 'ddgs' package. Install via: "
    "uv sync --extra web  (or: pip install ddgs)"
)


class WebSearchParams(BaseModel):
    query: str = Field(
        min_length=1,
        description="Search query (2-6 keywords optimal).",
    )
    max_results: int = Field(
        default=5, ge=1, le=20,
        description="How many hits to return (1-20).",
    )


def _preview(args: dict[str, Any]) -> str:
    return f"query: {args.get('query', '')}"


def _ddgs_search(query: str, max_results: int) -> list[dict[str, Any]]:
    assert DDGS is not None
    rows = DDGS().text(query, max_results=max_results)
    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        normalized.append(
            {
                "title": row.get("title", "") or "",
                "url": row.get("href", "") or "",
                "snippet": row.get("body", "") or "",
            }
        )
    return normalized


class WebSearch(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "web_search"
    description: str = (
        "Search the web. Returns a list of {title, url, snippet}. "
        "Use 2-6 keywords; prefer specific terms. Follow up with "
        "web_fetch on a hit's url to read its contents."
    )
    args_schema: type[BaseModel] = WebSearchParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=exact_match_on("query"),
        args_preview=_preview,
        timeout_sec=30.0,
        capability_flags=frozenset({"search_command"}),
    )
    config: WebSearchConfig | None = None

    def _run(self, query: str, max_results: int = 5) -> dict[str, Any]:
        raise NotImplementedError("web_search is async-only; use ainvoke")

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        # Config's max_results only wins when the caller didn't pass an explicit value.
        effective_max = max_results
        if (
            self.config is not None
            and max_results == WebSearchParams.model_fields["max_results"].default
            and self.config.max_results != effective_max
        ):
            effective_max = self.config.max_results

        return await self._search_duckduckgo(query, effective_max)

    async def _search_duckduckgo(
        self, query: str, max_results: int,
    ) -> dict[str, Any]:
        if not _HAS_DDGS:
            raise ToolError(_INSTALL_HINT)
        try:
            results = await asyncio.to_thread(_ddgs_search, query, max_results)
        except RatelimitException as exc:
            raise ToolError(
                f"web_search rate-limited by DuckDuckGo, try again shortly: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"web_search failed: {type(exc).__name__}: {exc}") from exc

        return {
            "provider": "duckduckgo",
            "query": query,
            "results": results,
        }
