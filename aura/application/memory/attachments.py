"""Envelope shape for attachment HumanMessages — CLI owns WHEN, this module owns SHAPE."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


def build_mcp_resource_message(
    *,
    server: str,
    uri: str,
    name: str,
    content: str,
) -> HumanMessage:
    """Wrap an MCP resource as ``<mcp-resource server uri[ name]>\\nbody\\n</mcp-resource>``.

    Empty ``content`` renders ``[empty resource]`` — empty-tag bodies look like
    model hallucinations. ``name`` equal to ``uri`` is dropped as redundant.
    """
    header = f'<mcp-resource server="{server}" uri="{uri}"'
    if name and name != uri:
        header += f' name="{name}"'
    header += ">"
    body = content if content else "[empty resource]"
    return HumanMessage(f"{header}\n{body}\n</mcp-resource>")
