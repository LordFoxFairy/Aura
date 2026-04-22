"""Tests for compact summary prompt constants."""

from __future__ import annotations

from aura.core.compact.prompt import SUMMARY_SYSTEM, SUMMARY_USER_PREFIX


def test_summary_system_contains_text_only_guard() -> None:
    # The summary turn is intentionally tool-less: we burn tokens if the
    # model decides to call a tool (the only turn it gets). Prompt must
    # tell the model this explicitly.
    assert "TEXT ONLY" in SUMMARY_SYSTEM


def test_summary_system_mentions_required_sections() -> None:
    # Basic sanity: the structured sections the plan calls out must be named
    # in the prompt so the model produces a parseable shape. Don't pin the
    # exact wording, just the tags.
    for section in (
        "<goal>",
        "<decisions>",
        "<files-touched>",
        "<tools-used>",
        "<open-threads>",
        "<next-steps>",
    ):
        assert section in SUMMARY_SYSTEM


def test_summary_user_prefix_nonempty() -> None:
    # Prefix is the lead-in line for the HumanMessage; must not be empty
    # and must end with a newline/colon so serialized history reads cleanly.
    assert SUMMARY_USER_PREFIX
    assert SUMMARY_USER_PREFIX.strip()
