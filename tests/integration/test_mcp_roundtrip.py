"""Integration: MCP manager discovery + resource reachability via the
``AgentSession.mcp_manager`` accessor.

The v0.10.x architecture exposes MCP resources via the CLI-layer
``@server:uri`` attachment preprocessor (see :mod:`cli.attachments`
and :file:`tests/integration/test_mcp_attachments.py`). There is no
LLM-tool surface for resource reads — ``aconnect`` exposes the live
manager and nothing more.

This file covers what the integration tier still needs to assert at the
manager-→-AgentSession boundary:

1. ``aconnect`` exposes the live :class:`MCPManager` on
   :attr:`AgentSession.mcp_manager` (the attachment preprocessor relies on this).
2. ``aconnect`` never auto-registers a ``mcp_read_resource`` tool
   regardless of whether the catalogue has entries (parity with
   claude-code).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aura.application import session as agent_module
from aura.application.runtime import mcp as runtime_mcp
from tests.conftest import FakeChatModel
from tests.integration.conftest import build_integration_agent


class FakeMCPManager:
    """Drop-in stand-in for :class:`aura.infrastructure.mcp.MCPManager`.

    Exposes exactly the surface :meth:`AgentSession.aconnect` touches:

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
        # below; ``__init__`` ignores it because AgentSession.aconnect calls
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
    ``resources``. Can't bind via partial because AgentSession calls the bare class.
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


@pytest.mark.asyncio
async def test_aconnect_exposes_manager_without_auto_registering_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catalogue has entries — prior versions would auto-register a
    # resource-reader tool here. The CLI @mention preprocessor replaces
    # that surface, so no such tool ever appears.
    resources = {
        "mem://a": "contents-of-a",
        "mem://b": "contents-of-b",
    }
    fake_cls = _make_manager_factory(resources)
    monkeypatch.setattr(agent_module, "MCPManager", fake_cls)
    monkeypatch.setattr(runtime_mcp, "MCPManager", fake_cls)

    from aura.application.session import AgentSession
    from aura.config.schema import AuraConfig
    from aura.infrastructure.persistence.storage import SessionStorage

    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = AgentSession(
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
        # B3: live MCP manager inside async loop → must use aclose().
        await agent.aclose()


@pytest.mark.asyncio
async def test_aconnect_empty_catalogue_still_exposes_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cls = _make_manager_factory({})
    monkeypatch.setattr(agent_module, "MCPManager", fake_cls)
    monkeypatch.setattr(runtime_mcp, "MCPManager", fake_cls)

    from aura.application.session import AgentSession
    from aura.config.schema import AuraConfig
    from aura.infrastructure.persistence.storage import SessionStorage

    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = AgentSession(
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
        # B3: live MCP manager inside async loop → must use aclose().
        await agent.aclose()


def _silence_unused(_a: Any = build_integration_agent) -> None:  # pragma: no cover
    return None
