"""Integration: ``@server:uri`` mentions → resource-body HumanMessage injection.

Exercises the CLI-layer ``@mention`` preprocessor (see
:mod:`aura.cli.attachments`) at three levels:

1. **Parser** — ``extract_mentions`` reads the regex shape that mirrors
   claude-code's ``extractMcpResourceMentions`` (attachments.ts:2792).
2. **Resolver** — ``extract_and_resolve_attachments`` hits a ``FakeMCPManager``
   for known URIs; unknowns are silently skipped + journalled (parity
   with claude-code's ``tengu_at_mention_mcp_resource_error`` branch).
3. **End-to-end** — the scripted REPL path (``Agent.astream(prompt,
   attachments=...)``) injects ``<mcp-resource>`` ``HumanMessage``\\ s into
   history BEFORE the user's turn, and the LLM's ``ainvoke`` receives
   them in its messages list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from aura.cli.attachments import (
    ResourceAttachment,
    extract_and_resolve_attachments,
    extract_mentions,
    render_attachments_as_messages,
)
from aura.config.schema import AuraConfig
from aura.core import agent as agent_module
from aura.core.agent import Agent
from aura.core.persistence.storage import SessionStorage
from tests.conftest import FakeChatModel, FakeTurn

# ---------------------------------------------------------------------------
# FakeMCPManager — satisfies the two accessors the preprocessor needs.
# ---------------------------------------------------------------------------


class FakeMCPManager:
    def __init__(
        self,
        configs: Any = None,
        *,
        resources: dict[str, str] | None = None,
        server_name: str = "srv",
    ) -> None:
        self._configs = configs
        self._server = server_name
        self._resources = resources or {}
        # Read call log, so tests can assert which URIs were fetched.
        self.read_calls: list[str] = []
        self.raise_on: str | None = None

    def resources_catalogue(
        self,
    ) -> list[tuple[str, str, str, str, str | None]]:
        return [
            (self._server, uri, uri.rsplit("/", 1)[-1] or uri, "", None)
            for uri in sorted(self._resources)
        ]

    async def read_resource(self, uri: str) -> dict[str, Any]:
        self.read_calls.append(uri)
        if self.raise_on is not None and uri == self.raise_on:
            raise RuntimeError(f"simulated read failure for {uri}")
        if uri not in self._resources:
            raise ValueError(f"unknown uri {uri!r}")
        return {
            "uri": uri,
            "server": self._server,
            "contents": [
                {"type": "text", "text": self._resources[uri], "uri": uri},
            ],
        }

    async def start_all(self) -> tuple[list[Any], list[Any]]:
        return [], []

    async def stop_all(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_extract_mentions_basic_shape() -> None:
    mentions = extract_mentions("summarize @srv:mem://doc.md for me")
    assert mentions == [("srv", "mem://doc.md")]


def test_extract_mentions_multiple_unique() -> None:
    mentions = extract_mentions(
        "compare @a:mem://x and @b:file:///tmp/y please"
    )
    assert mentions == [("a", "mem://x"), ("b", "file:///tmp/y")]


def test_extract_mentions_deduplicated() -> None:
    # Second occurrence of the same (server, uri) is not re-emitted —
    # matches claude-code's ``uniq(...)`` on the match list.
    mentions = extract_mentions("@s:u and again @s:u plus @s:u")
    assert mentions == [("s", "u")]


def test_extract_mentions_ignores_midword_at() -> None:
    # Parity with claude-code's ``(^|\s)@`` anchor: no prefix word char.
    assert extract_mentions("email user@host:80 was sent") == []


def test_extract_mentions_requires_colon_separator() -> None:
    # Bare ``@foo`` without ``:<uri>`` doesn't match (agent/file mentions
    # use different syntax and are not our concern here).
    assert extract_mentions("ping @someone about it") == []


def test_extract_mentions_on_empty_prompt() -> None:
    assert extract_mentions("") == []
    assert extract_mentions("   just plain text   ") == []


# ---------------------------------------------------------------------------
# Resolver — happy path, unknown, read failure, no manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_returns_attachment_for_known_mention() -> None:
    manager = FakeMCPManager(resources={"mem://doc.md": "THE DOC BODY"})
    prompt = "summarize @srv:mem://doc.md for me"
    resolved_prompt, attachments = await extract_and_resolve_attachments(
        prompt, manager,  # type: ignore[arg-type]
    )
    # Prompt is NOT rewritten (claude-code parity — attachments.ts only
    # reads + attaches; the @mention stays in the user's text).
    assert resolved_prompt == prompt
    assert len(attachments) == 1
    att = attachments[0]
    assert att.server == "srv"
    assert att.uri == "mem://doc.md"
    assert att.content == "THE DOC BODY"
    assert manager.read_calls == ["mem://doc.md"]


@pytest.mark.asyncio
async def test_resolve_skips_unknown_mention_silently() -> None:
    manager = FakeMCPManager(resources={"mem://doc.md": "body"})
    prompt = "consult @srv:mem://missing and @srv:mem://doc.md"
    resolved_prompt, attachments = await extract_and_resolve_attachments(
        prompt, manager,  # type: ignore[arg-type]
    )
    assert resolved_prompt == prompt
    # Only the known URI survived; the unknown was silently dropped (no
    # exception) — matches claude-code's ``return null`` path.
    assert [a.uri for a in attachments] == ["mem://doc.md"]
    # Manager was only queried for the known URI — catalogue short-circuits
    # the unknown before read_resource is ever called.
    assert manager.read_calls == ["mem://doc.md"]


@pytest.mark.asyncio
async def test_resolve_skips_mention_when_read_raises() -> None:
    manager = FakeMCPManager(resources={"mem://doc.md": "body"})
    manager.raise_on = "mem://doc.md"
    _, attachments = await extract_and_resolve_attachments(
        "please fetch @srv:mem://doc.md",
        manager,  # type: ignore[arg-type]
    )
    assert attachments == []
    # Read was attempted, just didn't succeed.
    assert manager.read_calls == ["mem://doc.md"]


@pytest.mark.asyncio
async def test_resolve_without_manager_returns_empty_attachments() -> None:
    prompt = "hello @srv:mem://doc.md world"
    resolved_prompt, attachments = await extract_and_resolve_attachments(
        prompt, None,
    )
    assert resolved_prompt == prompt
    assert attachments == []


@pytest.mark.asyncio
async def test_resolve_no_mentions_skips_manager_entirely() -> None:
    manager = FakeMCPManager(resources={"mem://doc.md": "body"})
    _, attachments = await extract_and_resolve_attachments(
        "just plain text", manager,  # type: ignore[arg-type]
    )
    assert attachments == []
    assert manager.read_calls == []


# ---------------------------------------------------------------------------
# Renderer — attachment → HumanMessage envelope
# ---------------------------------------------------------------------------


def test_render_produces_envelope_with_body() -> None:
    att = ResourceAttachment(
        server="srv",
        uri="mem://doc.md",
        name="doc.md",
        description=None,
        content="hello body",
    )
    [msg] = render_attachments_as_messages([att])
    assert isinstance(msg, HumanMessage)
    text = str(msg.content)
    assert '<mcp-resource server="srv" uri="mem://doc.md"' in text
    assert "hello body" in text
    assert text.endswith("</mcp-resource>")


def test_render_empty_list_returns_empty() -> None:
    assert render_attachments_as_messages([]) == []


# ---------------------------------------------------------------------------
# End-to-end via Agent.astream — the scripted LLM sees the attachment
# ---------------------------------------------------------------------------


def _cfg_with_one_server() -> dict[str, Any]:
    return {
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": []},
        "mcp_servers": [
            {
                "name": "srv",
                "transport": "stdio",
                "command": "echo",
                "args": ["noop"],
            }
        ],
    }


class _CaptureModel(FakeChatModel):
    """FakeChatModel that snoops the HumanMessages sent to ``_agenerate``."""

    def __init__(self, turns: list[FakeTurn]) -> None:
        super().__init__(turns=turns)
        self.__dict__["captured_messages"] = []

    @property
    def captured_messages(self) -> list[list[BaseMessage]]:
        return self.__dict__["captured_messages"]  # type: ignore[no-any-return]

    async def _agenerate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> object:
        # Snapshot the list so the assertion sees a stable view even if
        # the loop later mutates history.
        self.__dict__["captured_messages"].append(list(messages))
        return await super()._agenerate(
            messages, stop, run_manager, **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_end_to_end_attachment_reaches_llm_as_humanmessage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub MCPManager factory so aconnect wires our FakeMCPManager.
    fake_mgr_instance: dict[str, FakeMCPManager] = {}

    class _BoundFake(FakeMCPManager):
        def __init__(self, configs: Any) -> None:
            super().__init__(
                configs,
                resources={"mem://doc.md": "HERE IS THE DOC BODY"},
                server_name="srv",
            )
            fake_mgr_instance["m"] = self

    monkeypatch.setattr(agent_module, "MCPManager", _BoundFake)

    model = _CaptureModel(
        turns=[FakeTurn(message=AIMessage(content="ack"))],
    )
    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = Agent(
        config=cfg,
        model=model,
        storage=SessionStorage(tmp_path / "aura.db"),
    )
    try:
        await agent.aconnect()
        # Drive the turn the way the REPL does: resolve mentions first,
        # then hand both the prompt and rendered attachments to astream.
        prompt = "please read @srv:mem://doc.md"
        resolved_prompt, attachments = await extract_and_resolve_attachments(
            prompt, agent.mcp_manager,
        )
        attachment_messages = render_attachments_as_messages(attachments)
        async for _ in agent.astream(
            resolved_prompt, attachments=attachment_messages or None,
        ):
            pass
    finally:
        agent.close()

    # The scripted LLM was invoked exactly once (one FakeTurn).
    assert len(model.captured_messages) == 1
    sent = model.captured_messages[0]
    # Exactly one <mcp-resource> envelope was prepended.
    envelopes = [
        m for m in sent
        if isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and "<mcp-resource" in m.content
    ]
    assert len(envelopes) == 1
    assert "HERE IS THE DOC BODY" in str(envelopes[0].content)
    # Order: envelope comes BEFORE the user's HumanMessage.
    envelope_idx = sent.index(envelopes[0])
    user_idx = next(
        i for i, m in enumerate(sent)
        if isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and "please read @srv:mem://doc.md" in m.content
    )
    assert envelope_idx < user_idx


@pytest.mark.asyncio
async def test_end_to_end_unknown_mention_produces_no_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoundFake(FakeMCPManager):
        def __init__(self, configs: Any) -> None:
            super().__init__(configs, resources={}, server_name="srv")

    monkeypatch.setattr(agent_module, "MCPManager", _BoundFake)

    model = _CaptureModel(
        turns=[FakeTurn(message=AIMessage(content="ack"))],
    )
    cfg = AuraConfig.model_validate(_cfg_with_one_server())
    agent = Agent(
        config=cfg,
        model=model,
        storage=SessionStorage(tmp_path / "aura.db"),
    )
    try:
        await agent.aconnect()
        prompt = "talk about @srv:mem://nope and plain text"
        resolved_prompt, attachments = await extract_and_resolve_attachments(
            prompt, agent.mcp_manager,
        )
        assert attachments == []
        async for _ in agent.astream(resolved_prompt, attachments=None):
            pass
    finally:
        agent.close()

    sent = model.captured_messages[0]
    # No <mcp-resource> envelope was prepended; the prompt text is still
    # surfaced verbatim to the model (mention included — user sees their
    # own ``@srv:mem://nope`` in their message).
    assert not any(
        isinstance(m, HumanMessage)
        and isinstance(m.content, str)
        and "<mcp-resource" in m.content
        for m in sent
    )


def _silence_unused(_m: Any = MagicMock) -> None:  # pragma: no cover
    # Sentinel — keep MagicMock import reachable so future refactors that
    # need a bare MagicMock don't re-add it as an import (linter otherwise
    # flags unused imports).
    return None
