"""mcp_read_resource — LLM-invocable tool that reads a URI-identified MCP resource.

.. deprecated:: 0.10.x
    **Not auto-registered** on Agent construction. Prefer the
    ``@server:uri`` attachment syntax in the user prompt (see
    :mod:`aura.cli.attachments`) — that matches claude-code's
    ``extractMcpResourceMentions`` surface and injects the resource
    body as pre-loaded context BEFORE the LLM turn runs, instead of
    making the model decide when to pull.

    This class is kept importable for **programmatic SDK users** who want
    LLM-driven resource reads (e.g. an agent that runs without a human in
    the loop and legitimately needs to choose URIs at runtime). In that
    case, construct the tool manually and pass it via
    ``available_tools=`` to :class:`aura.core.agent.Agent`. The v0.10.x
    default Agent never wires it on its own.

MCP servers expose three kinds of capabilities: *tools*, *prompts*, and
*resources*. Tools are already reachable via the usual LangChain tool
surface (see :func:`aura.core.mcp.adapter.add_aura_metadata`); prompts
are bridged as commands (see :func:`aura.core.mcp.adapter.make_mcp_command`);
resources — text/blob content addressed by a stable URI — flow through
the CLI's ``@mention`` preprocessor today, leaving this tool as a legacy
opt-in surface.

Design choices worth defending:

- **Single tool, dynamic description.** We don't dynamically create one
  tool-per-resource: (a) servers can expose dozens of resources and each
  would steal a tool-schema slot from the provider's function-calling
  budget, (b) the catalogue may grow/shrink between sessions and
  registering tools at discovery time would cache-bust the pinned prompt
  prefix. Instead the tool's ``description`` carries the catalogue as
  text — the LLM sees "these URIs exist" alongside the tool itself, and
  a single slot covers the whole server fleet.

- **Read-only + concurrency-safe.** Reading a resource is pure lookup on
  the server side: no filesystem writes, no state mutation. Multiple
  LLM-parallel calls against distinct URIs can safely run under
  ``asyncio.gather`` without serialization.

- **Closure injection.** The tool takes a ``resource_reader`` coroutine
  at construction, matching the ``recorder`` / ``mode_setter`` pattern
  used by ``SkillTool`` / ``ExitPlanMode``. The closure proxies to
  :meth:`MCPManager.read_resource` so the tool never holds a reference
  to the manager directly.

If the LLM passes an unknown URI, we raise ``ToolError`` with the full
list of known URIs in the message — the model can self-correct next
turn without us needing a separate "list_resources" tool.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.schemas.tool import ToolError, tool_metadata

# The injected coroutine signature — takes a URI string, returns the
# normalized ``{uri, server, contents}`` dict built by
# :meth:`MCPManager.read_resource`. Bare ``Callable`` alias (not Protocol)
# so pydantic doesn't try to validate its internals.
ResourceReader = Callable[[str], Awaitable[dict[str, Any]]]


class ReadResourceParams(BaseModel):
    uri: str = Field(
        ...,
        min_length=1,
        description=(
            "The resource URI to read. Must match one of the URIs listed "
            "in this tool's description (the catalogue of known resources "
            "across all connected MCP servers). Examples: "
            "'file:///path/to/doc.md', 'db://snapshot/2026-04-23', or any "
            "custom scheme the server registers."
        ),
    )


def _preview(args: dict[str, Any]) -> str:
    return f"resource: {args.get('uri', '')}"


_BASE_DESCRIPTION = (
    "Read a resource from a connected MCP server by URI. Returns the "
    "resource contents as a list of {type, text|size, mime, uri} entries — "
    "text resources include the full body; blob resources return metadata "
    "only (size + mime) to avoid flooding the model with binary data."
)
_NO_RESOURCES_SUFFIX = "\n\nNo MCP resources are currently available."
_CATALOGUE_HEADER = "\n\nAvailable resources:\n"


def build_description(
    catalogue: list[tuple[str, str, str, str, str | None]],
) -> str:
    """Compose the tool's description from the resources catalogue.

    ``catalogue`` is the ``(server, uri, name, description, mime)`` list
    returned by :meth:`MCPManager.resources_catalogue`. We deliberately
    format each entry on its own line with the URI first — the LLM's
    function-calling implementation commonly scans the first ~200 chars
    of a description when ranking tools, so front-loading the URI keeps
    the most useful information visible even under truncation.

    Returned verbatim as the tool's ``description`` field; the pinned
    prefix cache key includes this string, so callers should compute it
    ONCE at Agent wiring time rather than per-turn.
    """
    if not catalogue:
        return _BASE_DESCRIPTION + _NO_RESOURCES_SUFFIX
    lines = [_BASE_DESCRIPTION + _CATALOGUE_HEADER]
    for server, uri, name, description, mime in catalogue:
        # Format: "- [server] uri — name: description (mime)"
        # Empty description / None mime are elided rather than printing
        # "None" — the LLM shouldn't see noise markers.
        parts = [f"- [{server}] {uri}"]
        if name and name != uri:
            parts.append(f" — {name}")
        if description:
            parts.append(f": {description}")
        if mime:
            parts.append(f" ({mime})")
        lines.append("".join(parts))
    return "\n".join(lines)


def _build_deprecated_metadata() -> dict[str, Any]:
    # Base metadata + a local ``deprecated`` flag. ``tool_metadata`` returns
    # a plain dict so we can tack on extra keys; no schema change needed on
    # the shared helper for a one-tool opt-in marker. SDK code that wants
    # to filter deprecated tools can check ``(tool.metadata or {}).get(
    # "deprecated") is True``.
    base = tool_metadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        max_result_size_chars=50_000,
        args_preview=_preview,
    )
    base["deprecated"] = True
    base["deprecated_since"] = "0.10.0"
    base["deprecated_replacement"] = (
        "@server:uri attachment syntax in the user prompt "
        "(see aura.cli.attachments)"
    )
    return base


class MCPReadResourceTool(BaseTool):
    """Read an MCP resource by URI.

    **Deprecated in v0.10.x.** Not auto-registered on Agent construction;
    prefer the ``@server:uri`` attachment syntax. Still importable for
    programmatic SDK users who want LLM-driven resource reads — pass the
    constructed instance via ``available_tools=`` to :class:`Agent`.

    Tool's description is built dynamically from the catalogue at
    wiring time — see :func:`build_description`.
    """

    # ``ResourceReader`` is a bare Callable alias, not a pydantic model —
    # same rationale as SkillTool's ``_recorder`` field.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "mcp_read_resource"
    description: str = _BASE_DESCRIPTION + _NO_RESOURCES_SUFFIX
    args_schema: type[BaseModel] = ReadResourceParams
    metadata: dict[str, Any] | None = _build_deprecated_metadata()
    _reader: ResourceReader = PrivateAttr()

    def __init__(
        self,
        *,
        resource_reader: ResourceReader,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseTool accepts ``description`` as a plain field override in
        # the constructor; forwarding it through ``kwargs`` means the
        # dynamically-built catalogue string lands on the instance. When
        # the caller didn't supply one we fall back to the class default
        # (empty catalogue). We DON'T stash it as a PrivateAttr because
        # LangChain reads ``description`` off the public attribute when
        # constructing the provider-facing tool schema.
        if description is not None:
            kwargs["description"] = description
        super().__init__(**kwargs)
        self._reader = resource_reader

    def _run(self, uri: str) -> dict[str, Any]:
        # MCPManager.read_resource is async — there's no sync path through
        # the MCP client library. Force callers through ainvoke, same
        # approach as ExitPlanMode / AskUserQuestion.
        raise NotImplementedError(
            "mcp_read_resource is async-only; use ainvoke"
        )

    async def _arun(self, uri: str) -> dict[str, Any]:
        try:
            return await self._reader(uri)
        except ValueError as exc:
            # Manager raises ValueError when the URI isn't in the
            # catalogue — convert to ToolError so the LLM sees a
            # structured failure (and the message lists known URIs).
            raise ToolError(str(exc)) from exc
