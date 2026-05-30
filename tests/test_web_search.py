"""Tests for aura.tools.web_search."""

from __future__ import annotations

from typing import Any

import pytest

from aura.config.schema import WebSearchConfig
from aura.domain.tool import ToolError
from aura.domain.tool_meta_access import meta_dict
from aura.tools.web_search import WebSearch


def test_web_search_tool_metadata_flags() -> None:
    tool = WebSearch(config=None)
    meta = meta_dict(tool)
    assert meta.get("is_read_only") is True
    assert meta.get("is_concurrency_safe") is True
    # max_result_size_chars is a legacy-only key not in ToolMetadata;
    # the budget hook uses the global cap when the per-tool cap is None.


def test_web_search_metadata_matcher_exact_on_query() -> None:
    tool = WebSearch(config=None)
    meta = meta_dict(tool)
    matcher = meta.get("rule_matcher")
    assert callable(matcher)
    # exact_match_on("query") carries a .key attribute per the matchers module convention.
    assert getattr(matcher, "key", None) == "query"
    assert matcher({"query": "langchain"}, "langchain") is True
    assert matcher({"query": "langchain"}, "something else") is False


def test_web_search_args_preview() -> None:
    tool = WebSearch(config=None)
    meta = meta_dict(tool)
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"query": "python typing"}) == "query: python typing"


class _FakeDDGS:
    """Drop-in replacement for ``ddgs.DDGS``.

    ``ddgs.DDGS().text(query, max_results=N)`` returns list[dict] with keys
    ``title`` / ``href`` / ``body``. We record every call so tests can assert
    max_results was forwarded.
    """

    calls: list[dict[str, Any]] = []  # class-level so tests can inspect after the fact

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        type(self).calls.append({"query": query, **kwargs})
        max_results = int(kwargs.get("max_results", 10))
        rows = [
            {
                "title": f"Result {i}",
                "href": f"https://example.com/{i}",
                "body": f"snippet {i}",
            }
            for i in range(max_results)
        ]
        return rows


async def test_web_search_with_mocked_ddgs_returns_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.tools.web_search as ws_mod

    _FakeDDGS.calls.clear()
    monkeypatch.setattr(ws_mod, "_HAS_DDGS", True)
    monkeypatch.setattr(ws_mod, "DDGS", _FakeDDGS, raising=False)

    tool = WebSearch(config=None)
    out = await tool._arun(query="python", max_results=3)

    assert out["provider"] == "duckduckgo"
    assert out["query"] == "python"
    assert isinstance(out["results"], list)
    assert len(out["results"]) == 3
    # All required keys are present on every hit.
    for hit in out["results"]:
        assert set(hit.keys()) == {"title", "url", "snippet"}


async def test_web_search_field_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """ddgs ``href`` → our ``url``; ddgs ``body`` → our ``snippet``."""
    import aura.tools.web_search as ws_mod

    class _Fixed:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            return [
                {"title": "T", "href": "https://x.example/1", "body": "B"},
            ]

    monkeypatch.setattr(ws_mod, "_HAS_DDGS", True)
    monkeypatch.setattr(ws_mod, "DDGS", _Fixed, raising=False)

    tool = WebSearch(config=None)
    out = await tool._arun(query="q", max_results=1)

    assert out["results"] == [
        {"title": "T", "url": "https://x.example/1", "snippet": "B"},
    ]


async def test_web_search_respects_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.tools.web_search as ws_mod

    _FakeDDGS.calls.clear()
    monkeypatch.setattr(ws_mod, "_HAS_DDGS", True)
    monkeypatch.setattr(ws_mod, "DDGS", _FakeDDGS, raising=False)

    tool = WebSearch(config=None)
    out = await tool._arun(query="python", max_results=7)

    assert len(out["results"]) == 7
    # max_results must be forwarded to the backend — not post-filtered here.
    assert _FakeDDGS.calls[-1]["max_results"] == 7


async def test_web_search_uses_config_max_results_when_param_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a WebSearchConfig provides a default max_results, it applies when
    the tool's own param uses the schema default (5). Explicit param overrides."""
    import aura.tools.web_search as ws_mod

    _FakeDDGS.calls.clear()
    monkeypatch.setattr(ws_mod, "_HAS_DDGS", True)
    monkeypatch.setattr(ws_mod, "DDGS", _FakeDDGS, raising=False)

    cfg = WebSearchConfig(provider="duckduckgo", max_results=9)
    tool = WebSearch(config=cfg)
    out = await tool._arun(query="python")  # use schema default 5 … config wins.
    assert len(out["results"]) == 9


async def test_web_search_ddgs_rate_limit_becomes_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ddgs.exceptions import RatelimitException

    import aura.tools.web_search as ws_mod

    class _RateLimited:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            raise RatelimitException("too many")

    monkeypatch.setattr(ws_mod, "_HAS_DDGS", True)
    monkeypatch.setattr(ws_mod, "DDGS", _RateLimited, raising=False)

    tool = WebSearch(config=None)
    with pytest.raises(ToolError, match="rate"):
        await tool._arun(query="python", max_results=3)


async def test_web_search_ddgs_not_installed_returns_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aura.tools.web_search as ws_mod

    monkeypatch.setattr(ws_mod, "_HAS_DDGS", False)

    tool = WebSearch(config=None)
    with pytest.raises(ToolError, match=r"uv sync --extra web"):
        await tool._arun(query="python", max_results=3)


def test_web_search_config_defaults() -> None:
    cfg = WebSearchConfig()
    assert cfg.provider == "duckduckgo"
    assert cfg.api_key_env is None
    assert cfg.max_results == 5


def test_web_search_config_rejects_unknown_provider() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WebSearchConfig(provider="bing")  # type: ignore[arg-type]  # deliberately off-type arg to exercise path


def test_web_search_is_registered_as_stateful() -> None:
    from aura.tools import BUILTIN_STATEFUL_TOOLS

    assert "web_search" in BUILTIN_STATEFUL_TOOLS
    assert BUILTIN_STATEFUL_TOOLS["web_search"] is WebSearch
