"""Tests for aura.core.permissions.safety — SafetyPolicy + is_protected."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from aura.core.permissions.safety import (
    DEFAULT_PROTECTED,
    DEFAULT_SAFETY,
    SafetyPolicy,
    is_protected,
)

# ---------------------------------------------------------------------------
# DEFAULT_PROTECTED + DEFAULT_SAFETY shape
# ---------------------------------------------------------------------------


def test_default_protected_is_a_tuple_of_strings() -> None:
    assert isinstance(DEFAULT_PROTECTED, tuple)
    assert all(isinstance(p, str) for p in DEFAULT_PROTECTED)
    # Spec §6 — all entries present (translated to pathspec-friendly globs).
    expected_entries = {
        "**/.git/**",
        "**/.aura/**",
        "**/.ssh/**",
        "~/.bashrc",
        "~/.zshrc",
        "~/.profile",
        "~/.bash_profile",
        "~/.zprofile",
        "/etc/**",
    }
    assert expected_entries <= set(DEFAULT_PROTECTED)


def test_default_safety_uses_default_protected_and_empty_exempt() -> None:
    assert DEFAULT_SAFETY.protected == DEFAULT_PROTECTED
    assert DEFAULT_SAFETY.exempt == ()


def test_safety_policy_is_frozen() -> None:
    policy = SafetyPolicy(protected=("**/.git/**",), exempt=())
    with pytest.raises((AttributeError, Exception)):
        policy.protected = ("changed",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Default entries — each blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_str",
    [
        "/some/project/.git/config",
        "/deep/nested/tree/.git/refs/heads/main",
        "/workspace/.aura/settings.json",
        "/home/alice/.ssh/id_rsa",
        "/etc/passwd",
        "/etc/nginx/nginx.conf",
    ],
)
def test_default_policy_blocks_protected_globs(path_str: str) -> None:
    assert is_protected(path_str, DEFAULT_SAFETY) is True


def test_default_policy_blocks_home_rc_files() -> None:
    home = Path.home()
    for rc in (".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"):
        target = home / rc
        assert is_protected(target, DEFAULT_SAFETY) is True, target


# ---------------------------------------------------------------------------
# Non-matching paths
# ---------------------------------------------------------------------------


def test_unprotected_path_returns_false() -> None:
    # An ordinary project file — nothing in DEFAULT_PROTECTED should match.
    assert is_protected("/Users/alice/projects/hello/src/main.py", DEFAULT_SAFETY) is False


def test_path_that_contains_git_as_substring_but_not_component_is_not_blocked() -> None:
    # "gitignore" contains "git" but isn't a .git/ dir.
    assert is_protected("/repo/docs/gitignore-primer.md", DEFAULT_SAFETY) is False


# ---------------------------------------------------------------------------
# Exempt overrides protected
# ---------------------------------------------------------------------------


def test_exempt_entry_overrides_protected() -> None:
    policy = SafetyPolicy(
        protected=("**/.git/**",),
        exempt=("**/.git/safe-config",),
    )
    assert is_protected("/repo/.git/safe-config", policy) is False
    # Other .git paths still blocked.
    assert is_protected("/repo/.git/config", policy) is True


def test_empty_exempt_does_not_override_anything() -> None:
    policy = SafetyPolicy(protected=("**/.git/**",), exempt=())
    assert is_protected("/repo/.git/config", policy) is True


def test_exempt_without_matching_protected_returns_false() -> None:
    # If protected doesn't match, the result is False regardless of exempt.
    policy = SafetyPolicy(protected=("**/.git/**",), exempt=("/tmp/*",))
    assert is_protected("/tmp/hello.txt", policy) is False


# ---------------------------------------------------------------------------
# Path normalization — relative paths, symlinks, tilde expansion
# ---------------------------------------------------------------------------


def test_relative_path_is_resolved_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set cwd to a tmp dir, then pass a relative path that lands in .git/.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / ".git").mkdir(parents=True)
    (tmp_path / "src" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert is_protected("./src/.git/HEAD", DEFAULT_SAFETY) is True


def test_symlink_into_protected_dir_is_blocked(tmp_path: Path) -> None:
    # Create a real .git/HEAD and a symlink to it outside .git/. is_protected
    # should follow resolve() and see the link target lives under .git/.
    real_git = tmp_path / ".git"
    real_git.mkdir()
    real_head = real_git / "HEAD"
    real_head.write_text("ref: refs/heads/main\n")
    link = tmp_path / "head-link"
    try:
        os.symlink(real_head, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/fs")
    assert is_protected(link, DEFAULT_SAFETY) is True


def test_nonexistent_path_does_not_crash(tmp_path: Path) -> None:
    # Path doesn't exist; resolve(strict=False) tolerates that and still
    # normalizes the string. is_protected must return a bool, not raise.
    ghost = tmp_path / "does" / "not" / "exist" / ".git" / "HEAD"
    assert is_protected(ghost, DEFAULT_SAFETY) is True


def test_home_rc_pattern_matches_expanded_tilde_target() -> None:
    # DEFAULT_PROTECTED contains "~/.bashrc" — the policy must expand that
    # pattern before matching so it matches an actual absolutized home path.
    target = Path.home() / ".bashrc"
    assert is_protected(target, DEFAULT_SAFETY) is True


def test_tilde_in_input_path_is_expanded() -> None:
    # Even if the caller hands us a literal "~/.bashrc" string, it should
    # match the expanded "~/.bashrc" pattern in DEFAULT_PROTECTED.
    assert is_protected("~/.bashrc", DEFAULT_SAFETY) is True


# ---------------------------------------------------------------------------
# Defensive — never crash
# ---------------------------------------------------------------------------


def test_none_input_returns_false_without_crashing() -> None:
    # Callers shouldn't pass None, but safety must never be the thing that
    # crashes the agent.
    assert is_protected(cast(Path, None), DEFAULT_SAFETY) is False


def test_empty_string_input_returns_false_without_crashing() -> None:
    assert is_protected("", DEFAULT_SAFETY) is False


def test_garbage_object_returns_false_without_crashing() -> None:
    # Non-Path, non-str input — defensive contract: return False, never raise.
    assert is_protected(cast(Path, 12345), DEFAULT_SAFETY) is False


def test_malformed_patterns_do_not_crash_is_protected() -> None:
    # pathspec can raise on malformed patterns; is_protected must still be
    # safe. Pick something pathspec accepts but use as a smoke test anyway.
    policy = SafetyPolicy(protected=("**/.git/**",), exempt=())
    assert isinstance(is_protected("/repo/.git/config", policy), bool)
