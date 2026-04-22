"""Aura's thin wrapper around ``langchain-mcp-adapters`` BaseTool / prompt objects.

Two public functions — no classes — because each one maps a plain library
object onto an Aura-native shape:

- :func:`add_aura_metadata` mutates a :class:`BaseTool` returned by
  :class:`MultiServerMCPClient` in place to attach Aura's capability flags
  + namespace its ``.name``. MCP servers do not self-declare destructive /
  concurrency-safe / size-bound semantics, so we default conservatively
  (destructive = True, concurrency-safe = False, size cap = 30k chars).
  The permission layer therefore prompts by default until the user adds
  a rule for the tool.

- :func:`make_mcp_command` wraps an MCP *prompt* as an Aura
  :class:`Command` so it can be registered into :class:`CommandRegistry`
  alongside built-ins and skill commands. ``source="mcp"`` lets ``/help``
  group it separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from aura.core.commands.types import CommandResult, CommandSource
from aura.schemas.tool import tool_metadata

if TYPE_CHECKING:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from aura.core.agent import Agent


_MCP_PREFIX = "mcp__"
_MAX_MCP_RESULT_CHARS = 30_000


def _args_preview(args: dict[str, Any]) -> str:
    """Generic one-line preview for MCP tool calls.

    MCP tools expose arbitrary JSON-Schema arg shapes, so we can't build a
    tool-specific preview (like bash(command=...)). The CLI permission
    prompt falls back to this when rendering an MCP tool call.
    """
    if not args:
        return "args: (none)"
    # Show keys only — values may be large or sensitive (auth tokens etc).
    return "args: " + ", ".join(sorted(args.keys()))


def add_aura_metadata(tool: BaseTool, *, server_name: str) -> BaseTool:
    """Mutate *tool* in-place: namespace the name and attach Aura metadata.

    Name is normalised to ``mcp__<server_name>__<original>``. If the library
    has already applied a prefix (tools coming back already namespaced), we
    detect the ``mcp__`` prefix and leave the name alone.

    Returns the same object for call-site convenience.
    """
    # StructuredTool.name / metadata are plain pydantic fields — mutating
    # them is supported by langchain-core. No reflection needed.
    if not tool.name.startswith(_MCP_PREFIX):
        tool.name = f"{_MCP_PREFIX}{server_name}__{tool.name}"
    tool.metadata = tool_metadata(
        is_read_only=False,
        is_destructive=True,
        is_concurrency_safe=False,
        max_result_size_chars=_MAX_MCP_RESULT_CHARS,
        rule_matcher=None,
        args_preview=_args_preview,
    )
    return tool


class _MCPPromptCommand:
    """Command that pulls an MCP prompt body on demand and prints it.

    We don't inject the body into Context on invocation — this keeps parity
    with SkillCommand's surface (``handled=True, kind="print"``). A later
    release can route prompt bodies through Context if MCP prompts become
    multi-turn templates instead of one-shot messages.
    """

    source: CommandSource = "mcp"

    def __init__(
        self,
        *,
        server_name: str,
        prompt_name: str,
        description: str,
        client: MultiServerMCPClient,
    ) -> None:
        self._server = server_name
        self._prompt = prompt_name
        self._client = client
        self.name = f"/{server_name}__{prompt_name}"
        self.description = description

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        from aura.core import journal

        try:
            messages = await self._client.get_prompt(self._server, self._prompt)
        except Exception as exc:  # noqa: BLE001
            # Server may have died between startup discovery and now. Surface
            # a user-visible failure + journal the reason rather than throw.
            journal.write(
                "mcp_prompt_fetch_failed",
                server=self._server,
                prompt=self._prompt,
                error=f"{type(exc).__name__}: {exc}",
            )
            return CommandResult(
                handled=True,
                kind="print",
                text=f"mcp prompt fetch failed: {exc}",
            )

        text = "\n".join(str(m.content) for m in messages)
        journal.write("mcp_prompt_invoked", server=self._server, prompt=self._prompt)
        return CommandResult(handled=True, kind="print", text=text)


def make_mcp_command(
    *,
    server_name: str,
    prompt_name: str,
    prompt_description: str,
    client: MultiServerMCPClient,
) -> _MCPPromptCommand:
    """Build a :class:`Command` that fetches and renders an MCP prompt."""
    return _MCPPromptCommand(
        server_name=server_name,
        prompt_name=prompt_name,
        description=prompt_description,
        client=client,
    )
