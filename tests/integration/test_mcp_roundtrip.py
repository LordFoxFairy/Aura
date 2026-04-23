"""Integration: MCP discovery → mcp_read_resource tool → LLM calls it.

Unit tests cover :class:`MCPManager` against a fake adapter, and
:class:`MCPReadResourceTool` against a synthetic catalogue. This tier
drives the whole pipeline through a real :class:`Agent`:

1. A stub ``MCPManager`` (monkeypatched into ``aura.core.agent``) reports
   2 resources at ``aconnect`` time.
2. The agent registers ``mcp_read_resource`` into the tool registry with
   the catalogue baked into its description.
3. The scripted LLM issues a ``mcp_read_resource(uri=...)`` call; the
   tool routes through the stub manager's ``read_resource`` and returns
   the contents.
4. Confirms the "no resources" path is clean — the tool is NOT
   registered when the catalogue is empty (matches the "no empty
   schemas" discipline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from aura.core import agent as agent_module
from aura.schemas.events import ToolCallCompleted
from tests.conftest import FakeChatModel, FakeTurn
from tests.integration.conftest import build_integration_agent, drain

# ---------------------------------------------------------------------------
# FakeMCPManager — drops in where aura.core.agent.MCPManager would go.
# ---------------------------------------------------------------------------


class FakeMCPManager:
    """Drop-in stand-in for :class:`aura.core.mcp.MCPManager`.

    Exposes exactly the surface :meth:`Agent.aconnect` touches:

    - Constructed with a configs list (ignored by the fake).
    - :meth:`start_all` → ``(tools, commands)``.
    - :meth:`resources_catalogue` → ``[(server, uri, name, desc, mime)...]``.
    - :meth:`read_resource` → normalised ``{uri, server, contents}`` dict.
    - :meth:`stop_all` → async no-op.
    """

    def __init__(
        self,
        configs: Any,
        *,
        resources: dict[str, str] | None = None,
    ) -> None:
        self._configs = configs
        # uri -> text body. Caller hands this in via the class factory
        # below; ``__init__`` ignores it because Agent.aconnect calls
        # ``MCPManager(self._config.mcp_servers)`` positionally.
        self._resources: dict[str, str] = resources or {}

    async def start_all(self) -> tuple[list[Any], list[Any]]:
        # No tools, no prompt commands — the resources-only code path is
        # what we care about in this test.
        return [], []

    def resources_catalogue(
        self,
    ) -> list[tuple[str, str, str, str, str | None]]:
        return [
            ("fake", uri, uri.rsplit("/", 1)[-1] or uri, "", None)
            for uri in sorted(self._resources)
        ]

    async def read_resource(self, uri: str) -> dict[str, Any]:
        if uri not in self._resources:
            raise ValueError(
                f"unknown MCP resource uri {uri!r}; "
                f"known uris: {sorted(self._resources)}"
            )
        return {
            "uri": uri,
            "server": "fake",
            "contents": [
                {"type": "text", "text": self._resources[uri], "uri": uri}
            ],
        }

    async def stop_all(self) -> None:
        return None


def _make_manager_factory(resources: dict[str, str]) -> type:
    """Build a class that looks like ``MCPManager(configs)`` but preloads
    ``resources``. Can't bind via partial because Agent calls the bare class.
    """

    class _BoundFake(FakeMCPManager):
        def __init__(self, configs: Any) -> None:
            super().__init__(configs, resources=resources)

    return _BoundFake


def _cfg_with_one_server() -> dict[str, Any]:
    return {
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["read_file"]},
        "mcp_servers": [
            {
                "name": "fake",
                "transport": "stdio",
                "command": "echo",
                "args": ["noop"],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test 1 — 2 resources → mcp_read_resource is registered with catalogue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aconnect_with_two_resources_registers_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = {
        "mem://a": "contents-of-a",
        "mem://b": "contents-of-b",
    }
    fake_cls = _make_manager_factory(resources)
    monkeypatch.setattr(agent_module, "MCPManager", fake_cls)

    # Agent must have mcp_servers configured; AuraConfig demands it exist.
    # We build manually so we can pass mcp_servers in.
    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage

    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "aura.db"),
    )
    try:
        await agent.aconnect()
        # Tool registered.
        assert "mcp_read_resource" in agent._available_tools
        tool = agent._available_tools["mcp_read_resource"]
        # Description carries both catalogued URIs.
        assert "mem://a" in tool.description
        assert "mem://b" in tool.description
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Test 2 — LLM calls mcp_read_resource → tool message carries content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_reads_resource_via_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = {"mem://a": "contents-of-a", "mem://b": "contents-of-b"}
    fake_cls = _make_manager_factory(resources)
    monkeypatch.setattr(agent_module, "MCPManager", fake_cls)

    turns = [
        FakeTurn(
            message=AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc_1",
                        "name": "mcp_read_resource",
                        "args": {"uri": "mem://a"},
                    }
                ],
            )
        ),
        FakeTurn(message=AIMessage(content="read it")),
    ]

    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage

    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=turns),
        storage=SessionStorage(tmp_path / "aura.db"),
    )
    try:
        await agent.aconnect()
        events = await drain(agent, "read A")
    finally:
        agent.close()

    completed = [e for e in events if isinstance(e, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].name == "mcp_read_resource"
    assert completed[0].error is None
    output = completed[0].output
    # Shape: {"uri":"mem://a","server":"fake","contents":[{"text":"contents-of-a", ...}]}
    assert isinstance(output, dict)
    assert output["uri"] == "mem://a"
    assert any(
        "contents-of-a" in str(c.get("text", "")) for c in output["contents"]
    )


# ---------------------------------------------------------------------------
# Test 3 — zero resources → mcp_read_resource NOT registered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_catalogue_does_not_register_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls = _make_manager_factory({})
    monkeypatch.setattr(agent_module, "MCPManager", fake_cls)

    from aura.config.schema import AuraConfig
    from aura.core.agent import Agent
    from aura.core.persistence.storage import SessionStorage

    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp_path / "aura.db"),
    )
    try:
        await agent.aconnect()
        # With zero resources, mcp_read_resource must NOT be registered —
        # matches the "empty catalogue, skip tool" discipline.
        assert "mcp_read_resource" not in agent._available_tools
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Sentinel: keep ``build_integration_agent`` imported so its public usage
# in this tier doesn't drift unnoticed. The test functions above construct
# Agent manually because they need to pass ``mcp_servers`` in the config —
# build_integration_agent doesn't expose that knob today.
# ---------------------------------------------------------------------------


def _silence_unused(_a: Any = build_integration_agent) -> None:  # pragma: no cover
    return None
