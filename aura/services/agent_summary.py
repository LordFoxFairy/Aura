"""Subagent terminal-output summarization (deterministic MVP).

Translates claude-code's ``services/AgentSummary/agentSummary.ts`` —
the "what did the subagent end up doing?" digest the parent's
``task_get`` tool returns instead of the full transcript.

Claude-code uses a cheap LLM call to produce the digest. Aura's MVP is
deterministic: extract the final assistant message, list the last
three tool-call names, and append cumulative token totals. Phase 4
can swap this for an LLM-driven path with the same call signature.

Cap and shape contract:

- Returned string is at most :data:`SUMMARY_CHAR_CAP` characters
  (currently 500). Tests assert this hard cap.
- Always includes a non-empty marker text — never returns an empty
  string even if the event list is empty (callers display the result
  verbatim and a blank string would render as "nothing here" which
  hides the fact that a subagent ran at all).
- Tool names appear in chronological order (oldest of the last 3
  first). Operators reading a summary expect "what just happened" to
  be at the tail, so the most recent activity reads naturally.

The function is sync + pure: no I/O, no model calls, no journal
writes. That keeps it safe to call from anywhere — including hot paths
like ``task_get`` that fire on every poll.
"""

from __future__ import annotations

from collections.abc import Iterable

from aura.schemas.events import AgentEvent, Final, ToolCallStarted

#: Hard cap on the returned summary string. 500 chars matches the
#: original spec; the value is the truncation budget the parent's
#: ``task_get`` tool can safely embed inline in a tool-result envelope
#: without dominating the LLM's context.
SUMMARY_CHAR_CAP: int = 500

#: How many trailing tool names to surface. 3 is the smallest number
#: that reliably captures "what the subagent ended up doing" without
#: dominating the budget — the LLM call that consumes this summary
#: usually only needs the most recent action.
TOOL_TAIL_COUNT: int = 3

#: Hard cap on the final-message excerpt before tool-name + token
#: tail are appended. Computed so a maxed-out tail leaves room for
#: the final-message slice without truncating mid-word.
_FINAL_MSG_BUDGET = 320


def summarize_subagent_run(
    events: Iterable[AgentEvent],
    *,
    model: str = "",  # noqa: ARG001 - reserved for future LLM-call variant
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> str:
    """Produce a deterministic ≤500-char digest of a subagent run.

    Pulls the final assistant message text + the last
    :data:`TOOL_TAIL_COUNT` tool-call names from the event stream and
    formats them into a single string. ``input_tokens`` /
    ``output_tokens`` (optional) are appended as a ``(in=N, out=M)``
    suffix so the parent's UI can render usage without a separate
    store lookup.

    ``model`` is currently unused — kept on the signature so the
    Phase 4 LLM-driven variant can be substituted in place without
    breaking call sites. (Mirrors claude-code's
    ``agentSummary(events, model)`` shape.)
    """
    final_text = ""
    tool_names: list[str] = []
    for event in events:
        if isinstance(event, ToolCallStarted):
            tool_names.append(event.name)
        elif isinstance(event, Final):
            final_text = event.message
    # Keep the *last* TOOL_TAIL_COUNT names in chronological order.
    tool_tail = tool_names[-TOOL_TAIL_COUNT:]
    # Build the assembled summary. Each piece is bounded so the final
    # result fits inside SUMMARY_CHAR_CAP without aggressive truncation
    # near a piece boundary.
    pieces: list[str] = []
    if final_text:
        excerpt = final_text.strip().replace("\n", " ")
        if len(excerpt) > _FINAL_MSG_BUDGET:
            excerpt = excerpt[: _FINAL_MSG_BUDGET - 1].rstrip() + "…"
        pieces.append(f"final: {excerpt}")
    else:
        pieces.append("final: (no assistant message)")
    if tool_tail:
        pieces.append("tools: " + ", ".join(tool_tail))
    if input_tokens or output_tokens:
        pieces.append(f"tokens: in={input_tokens}, out={output_tokens}")
    summary = " | ".join(pieces)
    # Final hard cap — belt-and-braces against an unusually large
    # final-message excerpt + worst-case tool-name lengths. Truncate
    # with an ellipsis so the consumer can tell the result was clipped.
    if len(summary) > SUMMARY_CHAR_CAP:
        summary = summary[: SUMMARY_CHAR_CAP - 1].rstrip() + "…"
    return summary


__all__ = ["SUMMARY_CHAR_CAP", "TOOL_TAIL_COUNT", "summarize_subagent_run"]
