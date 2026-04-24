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
# Unused in 0.4.0 — reserved for the next iteration where we restore
# "recently read" file bodies. Constants declared now so tests can pin them.
MAX_FILES_TO_RESTORE = 5
MAX_TOKENS_PER_FILE = 5_000

# Budget for the summary turn itself. Advisory — the FakeChatModel in tests
# doesn't enforce this; production models respect prompt guidance.
SUMMARY_BUDGET = 50_000

# Auto-compact trigger: when LoopState.total_tokens_used exceeds this value,
# a post_model observer would mark a pending compact. Wired as a config
# field in a later release (0.4.1). 0 = disabled.
AUTO_COMPACT_THRESHOLD = 150_000

# --- Microcompact (G2, v0.12) -----------------------------------------------
# Per-turn prompt-view compression of old tool_use/tool_result pair payloads.
# These knobs tune the *view transform* applied between ``Context.build`` and
# ``self._bound.ainvoke`` in the loop; they do NOT mutate stored history.
#
# Pair-count trigger: once the outgoing prompt contains more than this many
# compactable tool_use/tool_result pairs, the oldest ones get their payloads
# replaced with ``MICROCOMPACT_CLEAR_MARKER``. Mirrors claude-code's default.
MICROCOMPACT_TRIGGER_PAIRS = 8

# Number of most-recent compactable pairs to keep raw when the trigger fires.
# Claude-code enforces ``Math.max(1, keep_recent)`` — we mirror that floor in
# ``select_clear_ids`` so a misconfiguration of 0 never clears *everything*.
MICROCOMPACT_KEEP_RECENT = 5

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
