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
