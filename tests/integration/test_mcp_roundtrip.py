"""Integration: MCP manager discovery + resource reachability via the
``Agent.mcp_manager`` accessor.

The v0.10.x architecture exposes MCP resources via the CLI-layer
``@server:uri`` attachment preprocessor (see :mod:`aura.cli.attachments`
and :file:`tests/integration/test_mcp_attachments.py`). The old LLM-tool
auto-registration (``mcp_read_resource``) was deprecated at the same
time — the tool class is still importable for programmatic SDK users
but is no longer wired into the default Agent.

This file covers what the integration tier still needs to assert at the
manager-→-Agent boundary:

1. ``aconnect`` exposes the live :class:`MCPManager` on
   :attr:`Agent.mcp_manager` (the attachment preprocessor relies on this).
2. ``aconnect`` no longer auto-registers ``mcp_read_resource`` regardless
   of whether the catalogue has entries (parity with claude-code).
3. Programmatic SDK users can still import + instantiate
   :class:`MCPReadResourceTool` and have it round-trip a URI through the
   manager (the opt-in path still works).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aura.core import agent as agent_module
from tests.conftest import FakeChatModel
from tests.integration.conftest import build_integration_agent

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
# Test 1 — aconnect exposes the manager and does NOT register the tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aconnect_exposes_manager_without_auto_registering_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catalogue has entries — prior versions would auto-register
    # ``mcp_read_resource`` here. v0.10.x replaces that with the CLI
    # @mention preprocessor, so the tool must NOT appear.
    resources = {
        "mem://a": "contents-of-a",
        "mem://b": "contents-of-b",
    }
    fake_cls = _make_manager_factory(resources)
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
        # Manager is on the accessor the @mention preprocessor uses.
        assert agent.mcp_manager is not None
        catalogue = agent.mcp_manager.resources_catalogue()
        assert {uri for _, uri, *_ in catalogue} == {"mem://a", "mem://b"}
        # Tool is NOT auto-registered.
        assert "mcp_read_resource" not in agent._available_tools
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Test 2 — empty catalogue: manager still exposed, tool still not registered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aconnect_empty_catalogue_still_exposes_manager(
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
        # Manager is still there (needed by the @mention preprocessor
        # even though its catalogue is empty).
        assert agent.mcp_manager is not None
        assert agent.mcp_manager.resources_catalogue() == []
        assert "mcp_read_resource" not in agent._available_tools
    finally:
        agent.close()


# ---------------------------------------------------------------------------
# Test 3 — programmatic SDK use: the deprecated tool still works when wired
# explicitly by a caller that wants LLM-driven resource reads.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deprecated_tool_still_works_for_programmatic_sdk_users() -> None:
    # The class is deprecated (not auto-registered) but remains importable
    # + invocable. This guard prevents accidentally deleting it during
    # later cleanups — SDK users who opted into the LLM-driven surface
    # should not silently lose it.
    from aura.tools.mcp_read_resource import MCPReadResourceTool, build_description

    async def reader(uri: str) -> dict[str, Any]:
        return {
            "uri": uri,
            "server": "fake",
            "contents": [{"type": "text", "text": f"body-of-{uri}", "uri": uri}],
        }

    tool = MCPReadResourceTool(
        resource_reader=reader,
        description=build_description(
            [("fake", "mem://doc", "doc", "", None)],
        ),
    )
    # Deprecation marker is on metadata, but the tool still functions.
    assert tool.metadata and tool.metadata.get("deprecated") is True
    out = await tool.ainvoke({"uri": "mem://doc"})
    assert out["uri"] == "mem://doc"
    assert out["contents"][0]["text"] == "body-of-mem://doc"


# ---------------------------------------------------------------------------
# Sentinel: keep ``build_integration_agent`` imported so its public usage
# in this tier doesn't drift unnoticed. The test functions above construct
# Agent manually because they need to pass ``mcp_servers`` in the config —
# build_integration_agent doesn't expose that knob today.
# ---------------------------------------------------------------------------


def _silence_unused(_a: Any = build_integration_agent) -> None:  # pragma: no cover
    return None
