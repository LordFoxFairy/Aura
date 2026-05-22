"""Phase 4 Task 1 — :class:`CompactionTrigger` enum + :class:`CompactConfig`.

Locks the trigger taxonomy and the default-value contract so subsequent
Tasks (which migrate the hardcoded constants in ``compact.py`` and
``microcompact.py`` to read from this config) can't silently drift the
defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.application.compact import CompactionTrigger
from aura.config.schema import AuraConfig, CompactConfig

# ---------------------------------------------------------------------------
# CompactionTrigger enum
# ---------------------------------------------------------------------------


def test_compaction_trigger_has_four_members() -> None:
    """Spec §3 — exactly four named triggers, no more, no less."""
    assert {t.name for t in CompactionTrigger} == {
        "microcompact",
        "reactive",
        "auto",
        "manual",
    }


def test_compaction_trigger_values_match_names() -> None:
    """Each member's value equals its name (StrEnum convention).

    The journal serializes triggers as plain strings; pinning value==name
    keeps wire format identical to the Python identifier so operators
    can grep ``"trigger": "reactive"`` without translation.
    """
    for trigger in CompactionTrigger:
        assert trigger.value == trigger.name


def test_compaction_trigger_is_str_subclass() -> None:
    """``StrEnum`` members must round-trip through ``str()`` cleanly.

    The Compactor methods accept the enum but the journal writes the
    string value. ``isinstance(trigger, str)`` lets call sites pass the
    enum to anything expecting a plain string without an explicit
    ``.value`` access.
    """
    assert isinstance(CompactionTrigger.microcompact, str)
    assert str(CompactionTrigger.reactive) == "reactive"


# ---------------------------------------------------------------------------
# CompactConfig defaults
# ---------------------------------------------------------------------------


def test_compact_config_defaults_match_legacy_constants() -> None:
    """Spec §4 — defaults match Phase 1-3 hardcoded constants.

    This is THE migration safety net: Task 2 + Task 4 swap call sites
    from ``MAX_FILES_TO_RESTORE`` etc. to ``cfg.compact.<field>``. If
    these defaults drift, the swap silently changes runtime behavior.
    """
    cfg = CompactConfig()

    # Auto-compact threshold buffer — claude-code's 13k headroom.
    assert cfg.auto_threshold_buffer_tokens == 13_000

    # File re-injection caps after summary block replaces middle history.
    # ``max_tokens_per_file`` matches the legacy ``MAX_TOKENS_PER_FILE``
    # constant (5_000) — preserves Phase 1-3 behavior across the migration.
    assert cfg.max_files_to_restore == 5
    assert cfg.max_tokens_per_file == 5_000

    # Summary serialization caps.
    assert cfg.max_summary_message_chars == 6_000
    assert cfg.max_summary_tool_args_chars == 2_000

    # Fallback excerpt cap when even one message is too large.
    assert cfg.fallback_summary_char_limit == 12_000

    # Recursion bound on summarize-split retry.
    assert cfg.max_summary_split_depth == 12

    # Auto-compact circuit breaker.
    assert cfg.max_consecutive_failures == 3

    # Microcompact pair trigger + keep-recent floor.
    assert cfg.microcompact_trigger_pairs == 5
    assert cfg.microcompact_keep_recent == 3

    # Time-based gap trigger — None = disabled by default.
    assert cfg.time_based_gap_threshold_minutes is None


def test_compact_config_field_count() -> None:
    """Spec §4 — exactly nine knob fields plus two microcompact + one time.

    Total of 11 user-facing fields: 8 hard caps/limits + 1 circuit
    breaker count + 2 microcompact + 1 time-based. Locking the count
    prevents accidental field deletion or addition without spec churn.
    """
    # NB: the spec narrative says "9 fields" because it groups the two
    # microcompact knobs + time-based gap as one cluster. The actual
    # pydantic field count is 11. Pin both in case the model is reshaped.
    assert len(CompactConfig.model_fields) == 11


def test_compact_config_rejects_unknown_field() -> None:
    """``extra='forbid'`` mirrors the rest of AuraConfig sub-blocks."""
    with pytest.raises(ValidationError):
        CompactConfig.model_validate({"bogus": 1})


def test_compact_config_rejects_negative_buffers() -> None:
    """All token / char caps are ``ge=0``; negative values are nonsense."""
    with pytest.raises(ValidationError):
        CompactConfig.model_validate({"max_files_to_restore": -1})
    with pytest.raises(ValidationError):
        CompactConfig.model_validate({"auto_threshold_buffer_tokens": -1})


def test_compact_config_rejects_zero_split_depth() -> None:
    """``max_summary_split_depth>=1`` — a depth of 0 would skip the retry
    loop entirely on first PromptTooLong, defeating its purpose."""
    with pytest.raises(ValidationError):
        CompactConfig.model_validate({"max_summary_split_depth": 0})


def test_compact_config_rejects_zero_consecutive_failures() -> None:
    """``max_consecutive_failures>=1`` — 0 would trip the breaker before
    any attempt ran, permanently disabling auto-compact on session start."""
    with pytest.raises(ValidationError):
        CompactConfig.model_validate({"max_consecutive_failures": 0})


def test_compact_config_rejects_zero_gap_minutes() -> None:
    """``time_based_gap_threshold_minutes`` is either ``None`` (disabled)
    or ``>=1`` (a real wall-clock gap). 0 minutes would fire on every
    turn — that's a microcompact bug, not a feature."""
    with pytest.raises(ValidationError):
        CompactConfig.model_validate({"time_based_gap_threshold_minutes": 0})


# ---------------------------------------------------------------------------
# AuraConfig integration
# ---------------------------------------------------------------------------


def test_aura_config_has_compact_block_with_defaults() -> None:
    """``AuraConfig.compact`` is a fresh :class:`CompactConfig` by default."""
    cfg = AuraConfig()
    assert isinstance(cfg.compact, CompactConfig)
    assert cfg.compact.auto_threshold_buffer_tokens == 13_000


def test_aura_config_accepts_compact_overrides() -> None:
    """Operators can pin individual fields via JSON config; unspecified
    fields keep their default."""
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "x", "protocol": "openai"}],
        "router": {"default": "x:m"},
        "compact": {
            "max_files_to_restore": 10,
            "time_based_gap_threshold_minutes": 30,
        },
    })
    assert cfg.compact.max_files_to_restore == 10
    assert cfg.compact.time_based_gap_threshold_minutes == 30
    # Unspecified fields keep defaults.
    assert cfg.compact.auto_threshold_buffer_tokens == 13_000
    assert cfg.compact.microcompact_trigger_pairs == 5
