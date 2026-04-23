"""web_search tool — query → list of hits.

Backend is DuckDuckGo (no API key, zero-config) via the ``ddgs`` package
(the maintained fork of ``duckduckgo-search``).

``ddgs`` is declared under the ``web`` extras group. If the package is
absent, the tool stays visible to the LLM but invocation raises a
``ToolError`` telling the user how to install it. Rationale: a missing
optional dependency is a configuration error, not a crash — surface it
through the normal tool-error channel.

Metadata:

- ``is_read_only=True``: unlike ``web_fetch``, ``web_search`` sends a short
  opaque query to one engine and receives a fixed-shape result list. It
  does not dereference attacker-chosen URLs and does not leak arbitrary
  context. Auto-approving it matches claude-code's classification.
- ``is_concurrency_safe=True``: no mutable state.
- ``max_result_size_chars=8_000``: the post-tool budget hook truncates
  pathological result payloads. Typical runs are well under this.
- ``rule_matcher=exact_match_on("query")``: lets users pin rules like
  ``web_search(aura langchain loop)`` when they want to auto-allow a
  specific query without blanket-allowing the tool.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from aura.config.schema import WebSearchConfig
from aura.core.permissions.matchers import exact_match_on
from aura.schemas.tool import ToolError, tool_metadata

# Optional dependency. ``DDGS`` and ``RatelimitException`` are imported at
# module load so test code can ``monkeypatch.setattr`` on the attribute
# directly; ``_HAS_DDGS`` tells the tool whether the import actually
# succeeded. A missing install is reported at call time — NOT at import —
# so ``from aura.tools.web_search import WebSearch`` never fails.
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
    """Blocking DDGS call. Call via ``asyncio.to_thread`` from ``_arun``."""
    assert DDGS is not None  # narrowed by _HAS_DDGS check at the call site
    rows = DDGS().text(query, max_results=max_results)
    # ddgs returns list[dict] with keys title / href / body. Normalize to
    # our common shape. Keep it defensive — an unknown row shape gets
    # sensible empty strings rather than a KeyError.
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
    """Search the web via a configurable backend (default: DuckDuckGo)."""

    # WebSearchConfig is a pydantic model itself — arbitrary_types_allowed is
    # unnecessary, but ConfigDict is kept for symmetry with the other
    # config-carrying tools in the codebase.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "web_search"
    description: str = (
        "Search the web. Returns a list of {title, url, snippet}. "
        "Use 2-6 keywords; prefer specific terms. Follow up with "
        "web_fetch on a hit's url to read its contents."
    )
    args_schema: type[BaseModel] = WebSearchParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=8_000,
        rule_matcher=exact_match_on("query"),
        args_preview=_preview,
        # DDGS has no explicit timeout knob; a 30s outer deadline stops a
        # throttled / hung query from freezing the turn.
        timeout_sec=30.0,
        is_search_command=True,
    )
    # Injected at Agent construction. None means "use defaults" (DuckDuckGo,
    # max_results=5). A non-None config pins the provider and may override
    # the default max_results cap.
    config: WebSearchConfig | None = None

    def _run(self, query: str, max_results: int = 5) -> dict[str, Any]:
        raise NotImplementedError("web_search is async-only; use ainvoke")

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
    ) -> dict[str, Any]:
        # The schema default for ``max_results`` is 5. If the caller did NOT
        # pass an explicit value (i.e. we see the default) AND the config
        # declares a different default, the config wins. Explicit argument
        # always overrides.
        effective_max = max_results
        if (
            self.config is not None
            and max_results == WebSearchParams.model_fields["max_results"].default
            and self.config.max_results != effective_max
        ):
            effective_max = self.config.max_results

        # The Literal on WebSearchConfig.provider narrows the only valid value
        # to "duckduckgo"; no branching needed.
        return await self._search_duckduckgo(query, effective_max)

    async def _search_duckduckgo(
        self, query: str, max_results: int,
    ) -> dict[str, Any]:
        if not _HAS_DDGS:
            raise ToolError(_INSTALL_HINT)
        try:
            results = await asyncio.to_thread(_ddgs_search, query, max_results)
        except RatelimitException as exc:
            # Surface the upstream throttling as a recoverable tool error so
            # the LLM can pivot (retry later, narrow the query) instead of
            # crashing the turn.
            raise ToolError(
                f"web_search rate-limited by DuckDuckGo, try again shortly: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001 — network libraries raise a zoo of types
            raise ToolError(f"web_search failed: {type(exc).__name__}: {exc}") from exc

        return {
            "provider": "duckduckgo",
            "query": query,
            "results": results,
        }
