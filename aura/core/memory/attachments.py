"""Envelope-builder helpers for attachment ``HumanMessage``\\ s.

Why this module exists
======================

The Memory subsystem's load-bearing invariant (see
:mod:`aura.core.memory` ``__init__`` docstring) is that
:meth:`aura.core.memory.context.Context.build` is the single site where
messages are assembled for the LLM. Attachments — ``@server:uri``
mentions resolved by :mod:`aura.cli.attachments` — are a legitimate
second injection channel (per the v0.11 G1 behaviour: user's turn
persists *with* attachments BEFORE the first ``model.ainvoke``), but
their envelope *layout* (the ``<mcp-resource server="..." uri="...">``
tag, the header/body separation, the placeholder for blob-only
resources) is part of the same wire-shape the LLM reads from the
Context.build'd pinned prefix.

If the CLI layer owned the envelope construction directly (as it used
to), "the shape of messages the LLM sees" would live in two places:
``Context.build`` for rules/skills/memory, and ``cli/attachments.py``
for resource injections. Drift between the two is cosmetic today but
the invariant is what keeps the prompt structurally uniform — the LLM
is trained on a single ``<tag attrs>body</tag>`` envelope family, and
every place that emits one must emit it with the same rules.

So: the CLI decides WHEN to build an attachment (ingress parsing,
MCP read, catalogue lookup). Memory decides the SHAPE
(``build_mcp_resource_message``). The CLI renderer just calls into
this module; the envelope rules live next to the rest of the Memory
subsystem's envelope rules.

This module is intentionally pure — no I/O, no logging, no mutable
state — so that the shape invariant is testable in isolation of the
MCP client, the journal, and the CLI regex parser.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage


def build_mcp_resource_message(
    *,
    server: str,
    uri: str,
    name: str,
    content: str,
) -> HumanMessage:
    """Wrap a resolved MCP resource as a ``<mcp-resource>`` ``HumanMessage``.

    The envelope mirrors the ``<nested-memory>`` / ``<rule>`` pattern
    used by :meth:`aura.core.memory.context.Context.build`: an XML-ish
    open/close tag with identifying attributes on the header, the raw
    body on its own line, and the closing tag on its own line. The LLM
    is already primed by the pinned prefix to read this envelope family,
    so reusing the shape keeps the outgoing prompt structurally uniform.

    Parameters
    ----------
    server:
        MCP server name — carried on the header so the LLM can correlate
        the attachment with the ``@server:uri`` mention that still sits
        in the user's prompt text (the CLI preprocessor does not strip
        the mention, matching claude-code's behaviour).
    uri:
        Resource URI. Also on the header — same correlation reason.
    name:
        Friendly label from the catalogue. If it equals ``uri``, it is
        omitted (pure redundancy); otherwise rendered as the third
        header attribute.
    content:
        Flattened body text (text entries concatenated with blank lines;
        blobs pre-flattened to ``[blob N bytes, mime]`` placeholders by
        the caller). Empty string is treated as "blob-only" and rendered
        as a ``[empty resource]`` placeholder line — an empty-tag body
        looks like a model hallucination and we do not want the LLM
        second-guessing whether the resource resolved.

    Returns
    -------
    HumanMessage
        Exactly: ``<mcp-resource server="S" uri="U"[ name="N"]>\\nBODY\\n</mcp-resource>``.
        Structure (3 newline-delimited lines) is covered by unit tests —
        downstream renderers split on ``\\n`` to separate header from body.
    """
    header = f'<mcp-resource server="{server}" uri="{uri}"'
    if name and name != uri:
        header += f' name="{name}"'
    header += ">"
    body = content if content else "[empty resource]"
    return HumanMessage(f"{header}\n{body}\n</mcp-resource>")
