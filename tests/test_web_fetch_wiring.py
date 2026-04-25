"""Round 4E + Round 5H: web_fetch singleton must be wired by Agent build.

Agent construction calls ``set_default_model_factory(make_summary_model_factory(...))``
once. After that, the module-level ``web_fetch`` singleton can be invoked
without raising the "no factory configured" ToolError.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

import aura.tools.web_fetch  # noqa: F401 — ensure submodule is imported
from aura.config.schema import AuraConfig
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from aura.tools.web_fetch import web_fetch
from tests.conftest import FakeChatModel, FakeTurn

wf_module = sys.modules["aura.tools.web_fetch"]


def _minimal_config(*, with_summary: bool = False) -> AuraConfig:
    cfg: dict[str, Any] = {
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file"]},
    }
    if with_summary:
        cfg["web_fetch"] = {"summary_model": "openai:gpt-4o-mini"}
    return AuraConfig.model_validate(cfg)


def _storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(tmp_path / "aura.db")


@pytest.fixture(autouse=True)
def _isolate_factory() -> Any:
    """Reset the module-level factory between tests.

    Other tests may build an Agent and leave the singleton wired to a
    FakeChatModel that's been GC'd; clear before + after each test.
    """
    wf_module.set_default_model_factory(None)
    yield
    wf_module.set_default_model_factory(None)


@pytest.mark.asyncio
async def test_web_fetch_singleton_works_after_agent_build(
    tmp_path: Path,
) -> None:
    """End-to-end: build Agent → ``web_fetch.ainvoke`` returns a summary.

    We monkeypatch the urllib fetch so the test stays offline, but the
    summary path goes through the real factory wired by Agent
    construction.
    """
    fake = FakeChatModel(turns=[FakeTurn(AIMessage(content="summary text"))])
    Agent(
        config=_minimal_config(),
        model=fake,
        storage=_storage(tmp_path),
    )
    # The factory is now wired — the singleton must NOT raise the
    # "no factory configured" ToolError.
    fake_fetch = {
        "url": "https://example.com",
        "status": 200,
        "content_type": "text/html",
        "content": "Example body",
        "truncated": False,
    }
    with patch("aura.tools.web_fetch._fetch", return_value=fake_fetch):
        result = await web_fetch.ainvoke({
            "url": "https://example.com",
            "prompt": "what does the page say?",
        })
    assert result["status"] == 200
    assert "summary" in result
    assert result.get("error") is None


@pytest.mark.asyncio
async def test_summary_uses_main_model_when_no_summary_config(
    tmp_path: Path,
) -> None:
    """Default fallback path: no ``web_fetch.summary_model`` set →
    factory yields the main model. Regression confirming
    ``make_summary_model_factory(cfg, main_model)`` returns the
    main_model when no summary is configured."""
    fake = FakeChatModel(
        turns=[FakeTurn(AIMessage(content="from main model"))],
    )
    Agent(
        config=_minimal_config(with_summary=False),
        model=fake,
        storage=_storage(tmp_path),
    )
    fake_fetch = {
        "url": "https://example.com",
        "status": 200,
        "content_type": "text/html",
        "content": "Example body",
        "truncated": False,
    }
    with patch("aura.tools.web_fetch._fetch", return_value=fake_fetch):
        result = await web_fetch.ainvoke({
            "url": "https://example.com",
            "prompt": "extract anything",
        })
    assert result.get("error") is None
    assert "from main model" in (result.get("summary") or "")
