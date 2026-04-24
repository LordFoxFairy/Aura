"""Parse ``@server:uri`` MCP-resource mentions out of a user prompt and
resolve them to attachment messages before the LLM sees the turn.

Mirrors claude-code's ``extractMcpResourceMentions`` +
``processMcpResourceAttachments`` (``src/utils/attachments.ts:1990-2061``,
``:2792-2800``): the user names resources inline with ``@server:uri``, the
CLI preprocessor reads each resource via the MCP client BEFORE the turn
reaches the model, and each resolved body is prepended to the turn as a
``HumanMessage`` so the LLM sees it as pre-loaded context instead of
picking a tool to fetch it.

Design choices worth defending:

- **User-driven, not LLM-driven.** Resource reading is a side-effectful
  operation scoped by the user's intent ("look at this doc", "consult that
  snapshot"). Exposing it as an LLM tool (the original ``mcp_read_resource``
  surface) inverted the control: the model had to invent URIs and choose
  when to pull. The ``@mention`` surface matches claude-code, matches file
  drag-and-drop semantics, and stops URIs from being silently re-pulled
  turn after turn.

- **Silent skip on unknown / unresolvable mentions.** claude-code journals
  ``tengu_at_mention_mcp_resource_error`` and returns null (see
  attachments.ts:2011-2013, :2018, :2027, :2047). We do the same: if the
  manager isn't wired, if the server name isn't in the catalogue, or if
  ``read_resource`` raises, the mention is dropped from the attachment
  list and a journal entry records the failure. The original prompt text
  is still passed through unchanged — the user sees "@srv:uri" in their
  own message, which is the cheapest "this didn't resolve" signal we can
  give without blocking the turn on a pop-up.

- **Mentions are NOT stripped from the prompt.** claude-code leaves the
  ``@server:uri`` token in the prompt text (the file only extracts and
  attaches; it never rewrites the user's input). We keep parity: the LLM
  sees both the reference in the user's sentence AND the injected
  ``<mcp-resource>`` block, which helps it correlate "summarize the doc I
  mentioned" with the attachment body.

- **Normalization.** ``MCPManager.read_resource`` returns
  ``{uri, server, contents: [{type, text|size, mime, uri}, ...]}``. We flatten
  the text entries into a single string per attachment (blobs become a
  ``[blob <size> bytes, <mime>]`` placeholder — a text model can't consume
  base64 anyway). This is pure display logic — the raw dict stays
  available on the ``ResourceAttachment`` if a future renderer wants it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from aura.core.memory.attachments import build_mcp_resource_message
from aura.core.persistence import journal

if TYPE_CHECKING:
    from aura.core.mcp import MCPManager


# Matches claude-code's ``/(^|\s)@([^\s]+:[^\s]+)\b/g``. In Python we use a
# lookbehind that refuses word-character prefixes so ``foo@bar:baz`` is
# NOT treated as a mention (matches JS's "start-of-line OR whitespace before
# @"). The server and URI halves are captured separately so we can validate
# each one independently — the server name has to be a sane identifier,
# but the URI may contain colons (scheme://path), slashes, dots, etc.
#
# The URI half is ``\S+`` (one or more non-whitespace chars). Whitespace
# terminates the mention; a trailing punctuation like ``.`` is intentionally
# kept as part of the URI — claude-code doesn't strip it either, and the
# catalogue lookup will simply miss (silent skip) if the URI has trailing
# junk.
_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_-]+):(\S+)")


@dataclass(frozen=True)
class ResourceAttachment:
    """One resolved ``@server:uri`` mention.

    ``content`` is the flattened text body (concatenated text entries from
    the normalized MCP contents list). Empty string when the resource was
    blob-only — we still keep the attachment in the list so the LLM sees
    the placeholder line; a model asked to "read @srv:img.png" should not
    silently lose the reference just because we can't surface the bytes.
    """

    server: str
    uri: str
    name: str
    description: str | None
    content: str


def extract_mentions(prompt: str) -> list[tuple[str, str]]:
    """Return de-duplicated ``(server, uri)`` pairs mentioned in ``prompt``.

    Order is preserved for the first occurrence of each pair — mirrors
    claude-code's ``uniq(matches.map(...))`` shape at attachments.ts:2799.
    Empty list when no mentions are present; the REPL uses that to avoid
    touching the manager at all on "normal" prompts.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for match in _MENTION_RE.finditer(prompt):
        pair = (match.group(1), match.group(2))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _flatten_contents(contents: Iterable[dict[str, object]]) -> str:
    """Flatten the normalized ``contents`` list into one display string.

    Text entries' ``text`` fields are concatenated with a blank line
    between them. Blob entries render as ``[blob <size> bytes, <mime>]``
    placeholders — the model can't consume base64, so we surface enough
    metadata for it to tell the user what came back without poisoning the
    prompt with a megabyte of base64.
    """
    pieces: list[str] = []
    for entry in contents:
        kind = entry.get("type")
        if kind == "text":
            text = entry.get("text")
            if isinstance(text, str):
                pieces.append(text)
        elif kind == "blob":
            size = entry.get("size", "?")
            mime = entry.get("mime") or "application/octet-stream"
            pieces.append(f"[blob {size} bytes, {mime}]")
        else:  # "unknown" / future shapes
            pieces.append(f"[{kind} resource content]")
    return "\n\n".join(pieces)


async def extract_and_resolve_attachments(
    prompt: str,
    manager: MCPManager | None,
) -> tuple[str, list[ResourceAttachment]]:
    """Resolve every ``@server:uri`` mention in ``prompt``.

    Returns ``(prompt_unchanged, attachments)``. The prompt is returned
    verbatim — we do NOT strip the mentions; the LLM sees both the
    user's reference and the injected attachment body (matches
    claude-code's attachments.ts behaviour, which only reads + attaches).

    Unknown mentions are skipped silently and journalled as
    ``at_mention_mcp_resource_error`` for later debugging. Absent manager
    (no MCP configured, or ``aconnect`` never ran) also short-circuits to
    an empty attachment list — the prompt flows through unchanged.
    """
    mentions = extract_mentions(prompt)
    if not mentions:
        return prompt, []
    if manager is None:
        # No MCP configured — every mention is by definition unresolvable.
        # Journal once per mention so an operator reading logs can see what
        # the user tried to reference.
        for server, uri in mentions:
            journal.write(
                "at_mention_mcp_resource_error",
                reason="no_manager",
                server=server,
                uri=uri,
            )
        return prompt, []

    catalogue = manager.resources_catalogue()
    # Build a fast lookup: (server, uri) -> (name, description, mime). The
    # catalogue is small (tens of entries in practice) so a dict is fine.
    catalogue_map: dict[tuple[str, str], tuple[str, str, str | None]] = {
        (server, uri): (name, description, mime)
        for server, uri, name, description, mime in catalogue
    }

    attachments: list[ResourceAttachment] = []
    for server, uri in mentions:
        meta = catalogue_map.get((server, uri))
        if meta is None:
            journal.write(
                "at_mention_mcp_resource_error",
                reason="unknown_mention",
                server=server,
                uri=uri,
            )
            continue
        name, description, _mime = meta
        try:
            result = await manager.read_resource(uri)
        except Exception as exc:  # noqa: BLE001
            # Match claude-code: any read failure is dropped silently from
            # the attachment list and journalled. Do NOT surface as a user
            # error — the user already knows their prompt didn't resolve
            # (no content shows up in the next turn).
            journal.write(
                "at_mention_mcp_resource_error",
                reason="read_failed",
                server=server,
                uri=uri,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        content = _flatten_contents(result.get("contents", []))
        attachments.append(
            ResourceAttachment(
                server=server,
                uri=uri,
                name=name,
                description=description or None,
                content=content,
            )
        )
        journal.write(
            "at_mention_mcp_resource_success",
            server=server,
            uri=uri,
        )
    return prompt, attachments


def render_attachments_as_messages(
    attachments: list[ResourceAttachment],
) -> list[HumanMessage]:
    """Wrap each resolved attachment as a ``<mcp-resource>`` HumanMessage.

    This is the CLI-side ingress renderer — it only decides WHICH
    attachments to emit and in what order. The actual envelope SHAPE
    (``<mcp-resource ...>\\nBODY\\n</mcp-resource>``) lives in
    :func:`aura.core.memory.attachments.build_mcp_resource_message` so
    that the Memory subsystem retains a single construction site for
    every injected-into-outgoing-prompt HumanMessage shape — the same
    invariant :meth:`aura.core.memory.context.Context.build` upholds for
    rules / skills / nested memory.

    Empty list → empty return: the REPL can always call this
    unconditionally without a size guard.
    """
    return [
        build_mcp_resource_message(
            server=att.server,
            uri=att.uri,
            name=att.name,
            content=att.content,
        )
        for att in attachments
    ]
