"""Unit tests for ``aura.core.memory.attachments`` — the envelope-builder
helpers that shape ``<mcp-resource>`` ``HumanMessage``\\ s.

The wire-shape of these messages is load-bearing: the LLM is trained on
claude-code's ``<mcp-resource>`` pattern, and Context.build's whole
walk-up / walk-down story depends on every injected-into-outgoing-prompt
HumanMessage shape being constructed *inside the memory subsystem*.
``cli/attachments.py`` is an ingress layer that decides WHEN to build an
attachment; the actual envelope lives here so Context owns the layout
invariant (memory/__init__.py docstring).

These tests pin the envelope shape independent of the CLI parser. They
are the first line of defense against a drift — if someone "optimizes"
the envelope and forgets the header attributes, or drops the trailing
``</mcp-resource>``, tests here fail before the integration tests even
get a chance.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from aura.core.memory.attachments import build_mcp_resource_message


def test_build_envelope_has_server_and_uri_attributes() -> None:
    """Header must carry ``server`` and ``uri`` — the two fields the LLM
    keys off to correlate the attachment with the ``@server:uri`` mention
    that still sits in the user's prompt text.
    """
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://doc.md",
        name="doc.md",
        content="hello body",
    )
    assert isinstance(msg, HumanMessage)
    text = str(msg.content)
    assert text.startswith('<mcp-resource server="srv" uri="mem://doc.md"')
    assert text.endswith("</mcp-resource>")
    assert "hello body" in text


def test_build_envelope_omits_name_when_equal_to_uri() -> None:
    """The CLI passes ``name=uri`` when the catalogue has no friendlier
    label. Emitting ``name="mem://doc"`` alongside ``uri="mem://doc"`` is
    pure redundancy — we drop it, matching the original
    ``render_attachments_as_messages`` code.
    """
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://doc",
        name="mem://doc",
        content="body",
    )
    text = str(msg.content)
    assert " name=" not in text.split("\n", 1)[0]


def test_build_envelope_includes_name_when_different() -> None:
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://abc123",
        name="Friendly Doc",
        content="body",
    )
    text = str(msg.content)
    header = text.split("\n", 1)[0]
    assert ' name="Friendly Doc"' in header


def test_build_envelope_empty_content_renders_placeholder() -> None:
    """A blob-only resource arrives with ``content=""``. We still want an
    envelope in the turn (so the LLM sees the reference was resolved);
    a placeholder line beats an empty tag that looks like a model
    hallucination.
    """
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://img.png",
        name="img.png",
        content="",
    )
    text = str(msg.content)
    assert "[empty resource]" in text
    assert text.endswith("</mcp-resource>")


def test_build_envelope_preserves_unicode_body() -> None:
    """Resource bodies can be any UTF-8 text — CJK, emoji, RTL. The
    envelope must not mangle them (no escape, no encode-decode round-trip
    that could lossy-normalize).
    """
    body = "中文 — привет — 🎉"
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://intl.txt",
        name="intl.txt",
        content=body,
    )
    assert body in str(msg.content)


def test_build_envelope_preserves_long_content_verbatim() -> None:
    """Large (MB-scale) resource bodies must pass through verbatim — we
    never truncate here. Truncation is a policy decision that belongs to
    whoever assembles the attachment list (the CLI preprocessor or a
    future size-guard), not to the envelope builder.
    """
    body = "x" * 100_000
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://big.txt",
        name="big.txt",
        content=body,
    )
    text = str(msg.content)
    assert body in text
    # Header + body + close tag. No silent truncation.
    assert len(text) >= 100_000


def test_build_envelope_body_newline_separation() -> None:
    """Header and body must be separated by exactly one ``\\n`` — the
    original code uses an f-string ``{header}\\n{body}\\n</mcp-resource>``.
    Downstream renderers (e.g., the compact / render-repr paths) rely on
    this to split header attributes from the raw body.
    """
    msg = build_mcp_resource_message(
        server="srv",
        uri="mem://doc",
        name="doc",
        content="BODY",
    )
    text = str(msg.content)
    # Exactly: ``<mcp-resource ...>\nBODY\n</mcp-resource>``
    lines = text.split("\n")
    assert len(lines) == 3
    assert lines[0].startswith("<mcp-resource ")
    assert lines[1] == "BODY"
    assert lines[2] == "</mcp-resource>"
