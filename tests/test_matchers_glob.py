"""Glob support in ``exact_match_on`` — back-compat + new behaviour.

Without glob metachars in the rule content, the matcher behaves as before
(verbatim equality). With ``*`` or ``?`` it switches to ``fnmatchcase`` so
a rule like ``bash(npm install *)`` covers every variant.

This closes the "settings.json allow doesn't persist across sessions" gap:
users were authoring exact-match rules for bash commands that varied
slightly each invocation (different cwd quoting, |tail -N suffix, etc.),
so the asker re-prompted every time. Glob support lets one rule cover the
whole family.
"""

from __future__ import annotations

from aura.core.permissions.matchers import exact_match_on


def test_exact_match_no_glob_meta_matches_verbatim() -> None:
    m = exact_match_on("command")
    assert m({"command": "npm test"}, "npm test") is True
    assert m({"command": "npm tests"}, "npm test") is False  # no fuzzy


def test_exact_match_no_glob_meta_rejects_substring() -> None:
    m = exact_match_on("command")
    assert m({"command": "echo hi; npm test"}, "npm test") is False


def test_exact_match_glob_star_matches_any_suffix() -> None:
    m = exact_match_on("command")
    assert m({"command": "npm install fastapi"}, "npm install *") is True
    # ``*`` requires at least one char after the literal space.
    assert m({"command": "npm install"}, "npm install *") is False


def test_exact_match_glob_star_alone_matches_anything_starting() -> None:
    m = exact_match_on("command")
    assert m({"command": "ls /tmp"}, "ls *") is True
    assert m({"command": "ls -la /tmp"}, "ls *") is True


def test_exact_match_glob_star_in_middle() -> None:
    m = exact_match_on("command")
    assert m({"command": "echo hello world"}, "echo * world") is True
    assert m({"command": "echo hello"}, "echo * world") is False


def test_exact_match_question_mark_single_char() -> None:
    m = exact_match_on("command")
    assert m({"command": "ls a"}, "ls ?") is True
    assert m({"command": "ls ab"}, "ls ?") is False


def test_exact_match_glob_doesnt_match_different_prefix() -> None:
    m = exact_match_on("command")
    assert m({"command": "rm /tmp/foo"}, "ls *") is False


def test_exact_match_path_descendant_glob() -> None:
    """``read_file`` glob: ``read_file(/tmp/*.log)`` should hit log files."""
    m = exact_match_on("path")
    assert m({"path": "/tmp/app.log"}, "/tmp/*.log") is True
    assert m({"path": "/tmp/app.txt"}, "/tmp/*.log") is False


def test_exact_match_missing_arg_returns_false() -> None:
    """Defensive: missing arg key → False, never raise."""
    m = exact_match_on("command")
    assert m({}, "anything") is False
    assert m({"other": "x"}, "anything *") is False


def test_exact_match_non_string_arg_returns_false() -> None:
    m = exact_match_on("command")
    assert m({"command": 42}, "*") is False
    assert m({"command": None}, "*") is False


def test_exact_match_glob_metachar_in_rule_with_no_actual_glob() -> None:
    """A literal ``*`` in the rule is interpreted as glob — that's the
    contract. If a user genuinely needs a literal ``*`` in their command
    they can author the rule WITHOUT a ``*`` and use a different framing,
    or use a path matcher. Documented limitation.
    """
    m = exact_match_on("command")
    # The rule "echo *" should match "echo " + anything.
    assert m({"command": "echo literal"}, "echo *") is True


def test_key_attribute_preserved_after_glob_extension() -> None:
    """``.key`` attribute survives — CLI rule_hint relies on it."""
    m = exact_match_on("command")
    assert getattr(m, "key", None) == "command"
