"""Compact thresholds.

All knobs live here so callers and tests agree on a single source of truth.
The docs/research/claude-code-design-principles.md note explains why each
default is what it is (in short: claude-code parity for the ones that
matter, engineering judgement for the rest — Aura has far fewer coupled
systems than claude-code so the scale is deliberately smaller).
"""

from __future__ import annotations

# Number of full turns (= HumanMessage/AIMessage pairs) preserved raw at the
# tail of history. Anything older becomes part of the summary block.
KEEP_LAST_N_TURNS = 3

# Per-run caps on the post-compact re-injection of recently touched files.
# Live — read by ``compact.py::_build_recent_files`` to cap the volume of
# file bodies re-added after the summary block replaces middle history.
MAX_FILES_TO_RESTORE = 5
MAX_TOKENS_PER_FILE = 5_000

# Auto-compact trigger: when LoopState.total_tokens_used exceeds this value,
# a post_model observer would mark a pending compact. Wired via the
# ``auto_compact_threshold`` constructor kwarg on ``Agent``. 0 = disabled.
#
# Sentinel value ``-1`` means "derive from current model's context window"
# via :func:`auto_compact_threshold_for`. F-0910-001: a fixed 150k value is
# model-blind — too low for 1M-context models (premature compaction) and
# too high for 32k-context models (overflow). The default has been changed
# to ``-1`` so a fresh Agent gets model-aware behaviour out of the box; an
# explicit positive value still overrides.
AUTO_COMPACT_THRESHOLD = -1

# F-0910-001: claude-code uses ``window - 13_000`` as its auto-compact
# trigger (``13k`` headroom for the next user turn + summary scratch). We
# mirror that constant exactly so threshold semantics match.
AUTO_COMPACT_HEADROOM_TOKENS = 13_000


def auto_compact_threshold_for(model_spec: str) -> int:
    """Return ``ctx_window - AUTO_COMPACT_HEADROOM_TOKENS`` for ``model_spec``.

    F-0910-001: previously the threshold was a hard-coded 150k constant —
    too low on 1M-context frontier models (premature compact) and too high
    on 32k-context models (overflow before trigger). This helper returns a
    model-aware floor; callers can still pass an explicit override via the
    ``auto_compact_threshold`` constructor kwarg.

    Floors at 1k so a misconfigured / unknown small-window model still
    yields a positive (enabled) threshold instead of <=0 ("disabled").
    """
    from aura.core.llm import get_context_window

    window = get_context_window(model_spec)
    threshold = window - AUTO_COMPACT_HEADROOM_TOKENS
    return max(1_000, threshold)

# --- Microcompact (G2, v0.12) -----------------------------------------------
# Per-turn prompt-view compression of old tool_use/tool_result pair payloads.
# These knobs tune the *view transform* applied between ``Context.build`` and
# ``self._bound.ainvoke`` in the loop; they do NOT mutate stored history.
#
# Pair-count trigger: once the outgoing prompt contains more than this many
# compactable tool_use/tool_result pairs, the oldest ones get their payloads
# replaced with ``MICROCOMPACT_CLEAR_MARKER``.
#
# Claude-code defaults to 8/5 — tuned for its higher average tool-call rate
# per turn. Aura turns are leaner (fewer concurrent tool calls, tighter
# prompts), so the pair accumulation rate is roughly half. 5/3 matches the
# same "keep ~60% of trigger" ratio at a threshold that actually fires in
# realistic Aura sessions instead of sitting above the practical ceiling.
MICROCOMPACT_TRIGGER_PAIRS = 5

# Number of most-recent compactable pairs to keep raw when the trigger fires.
# Claude-code enforces ``Math.max(1, keep_recent)`` — we mirror that floor in
# ``select_clear_ids`` so a misconfiguration of 0 never clears *everything*.
# Invariant: must be strictly less than ``MICROCOMPACT_TRIGGER_PAIRS`` so the
# trigger can actually fire clears once crossed (see Agent.__init__ guard).
MICROCOMPACT_KEEP_RECENT = 3

# Literal string that replaces the old ToolMessage.content payload. Chosen to
# be unambiguous + compact; claude-code uses the same wording.
MICROCOMPACT_CLEAR_MARKER = "[Old tool result content cleared]"

# Allowlist of tool names whose pair payloads are eligible for compression.
# Rationale:
#   - INCLUDE: high-volume / low-signal I/O — file read/write/edit, shell
#     (bash + bash_background), grep, glob, web_fetch, web_search. The model
#     rarely needs to re-read these mid-session; a one-line marker is enough
#     context to know the call happened.
#   - EXCLUDE: high-signal / low-volume — subagent lifecycle (task_*),
#     todo_write (authoritative state), skill (mid-turn instruction
#     injection), ask_user_question (user text), plan-mode transitions,
#     MCP reads (user-facing resources). These carry small payloads whose
#     content materially affects subsequent reasoning, and compressing them
#     corrupts the model's view of session state.
# Names here must exactly match each tool's ``BaseTool.name`` attribute —
# see ``aura/tools/{read_file,write_file,edit_file,bash,bash_background,
# grep,glob,web_fetch,web_search}.py``.
MICROCOMPACT_COMPACTABLE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "bash",
    "bash_background",
    "grep",
    "glob",
    "web_fetch",
    "web_search",
})
