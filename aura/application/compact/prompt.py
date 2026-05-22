"""Summary turn prompts — kept as module constants so tests can pin them."""

from __future__ import annotations

SUMMARY_SYSTEM = """You are compressing a conversation history.

CRITICAL: Respond with TEXT ONLY. Do NOT call any tools. Tool calls
will be discarded and you will have wasted your only turn.

Structure your response with these sections, in order:
- <goal>: what the user was trying to accomplish
- <decisions>: key decisions made and their rationale
- <files-touched>: files read/written/edited with 1-line per file
- <tools-used>: tools invoked and their outcomes (one-line each)
- <open-threads>: incomplete items, questions the user raised that
  weren't answered, errors encountered
- <next-steps>: what was planned to happen next

Keep each section concise. Budget: 4000 tokens total.
"""

SUMMARY_USER_PREFIX = "Summarize the following conversation history:\n\n"
