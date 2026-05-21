"""mcp_read_resource — read a URI-identified MCP resource."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from aura.schemas.tool import ToolError, ToolMetadata

ResourceReader = Callable[[str], Awaitable[dict[str, Any]]]


class ReadResourceParams(BaseModel):
    uri: str = Field(
        ...,
        min_length=1,
        description="Resource URI; must match one of the URIs in the tool description.",
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
    if not catalogue:
        return _BASE_DESCRIPTION + _NO_RESOURCES_SUFFIX
    lines = [_BASE_DESCRIPTION + _CATALOGUE_HEADER]
    for server, uri, name, description, mime in catalogue:
        parts = [f"- [{server}] {uri}"]
        if name and name != uri:
            parts.append(f" — {name}")
        if description:
            parts.append(f": {description}")
        if mime:
            parts.append(f" ({mime})")
        lines.append("".join(parts))
    return "\n".join(lines)




class MCPReadResourceTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "mcp_read_resource"
    description: str = _BASE_DESCRIPTION + _NO_RESOURCES_SUFFIX
    args_schema: type[BaseModel] = ReadResourceParams
    aura_metadata: ToolMetadata = ToolMetadata(
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        rule_matcher=None,
        args_preview=_preview,
        timeout_sec=None,
        capability_flags=frozenset({"deprecated"}),
    )
    _reader: ResourceReader = PrivateAttr()

    def __init__(
        self,
        *,
        resource_reader: ResourceReader,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        if description is not None:
            kwargs["description"] = description
        super().__init__(**kwargs)
        self._reader = resource_reader

    def _run(self, uri: str) -> dict[str, Any]:
        raise NotImplementedError(
            "mcp_read_resource is async-only; use ainvoke"
        )

    async def _arun(self, uri: str) -> dict[str, Any]:
        try:
            return await self._reader(uri)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
