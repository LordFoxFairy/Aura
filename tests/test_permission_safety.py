"""Tests for aura.core.permissions.safety — SafetyPolicy + is_protected.

Post Plan B refactor (2026-04-21): safety is direction-aware. Two lists,
one ``is_write`` kwarg. Writes block destructive tools; reads block any
path-arg tool and use a narrower set of globs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pytest

from aura.core.permissions.safety import (
    DEFAULT_PROTECTED_READS,
    DEFAULT_PROTECTED_WRITES,
    DEFAULT_SAFETY,
    SafetyPolicy,
    is_protected,
)

# ---------------------------------------------------------------------------
# DEFAULT_PROTECTED_* + DEFAULT_SAFETY shape
# ---------------------------------------------------------------------------


def test_default_protected_writes_is_a_tuple_of_strings() -> None:
    assert isinstance(DEFAULT_PROTECTED_WRITES, tuple)
    assert all(isinstance(p, str) for p in DEFAULT_PROTECTED_WRITES)
    # Spec §6 writes list — all entries present.
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
    assert expected_entries <= set(DEFAULT_PROTECTED_WRITES)


def test_default_protected_reads_is_a_tuple_of_strings() -> None:
    assert isinstance(DEFAULT_PROTECTED_READS, tuple)
    assert all(isinstance(p, str) for p in DEFAULT_PROTECTED_READS)
    # Spec §6 reads list — narrower than writes.
    expected_entries = {
        "**/.ssh/**",
        "~/.bashrc",
        "~/.zshrc",
        "~/.profile",
        "~/.bash_profile",
        "~/.zprofile",
        "/etc/**",
    }
    assert expected_entries <= set(DEFAULT_PROTECTED_READS)


def test_reads_and_writes_are_independent_tuples() -> None:
    # Independent identity (not the same object) so a future mutation to
    # one can't silently alter the other.
    assert DEFAULT_PROTECTED_READS is not DEFAULT_PROTECTED_WRITES
    # And ``.git/`` / ``.aura/`` appear ONLY on writes — the whole point of
    # the split.
    assert "**/.git/**" in DEFAULT_PROTECTED_WRITES
    assert "**/.git/**" not in DEFAULT_PROTECTED_READS
    assert "**/.aura/**" in DEFAULT_PROTECTED_WRITES
    assert "**/.aura/**" not in DEFAULT_PROTECTED_READS


def test_default_safety_uses_default_lists_and_empty_exempt() -> None:
    assert DEFAULT_SAFETY.protected_writes == DEFAULT_PROTECTED_WRITES
    assert DEFAULT_SAFETY.protected_reads == DEFAULT_PROTECTED_READS
    assert DEFAULT_SAFETY.exempt == ()


def test_safety_policy_is_frozen() -> None:
    policy = SafetyPolicy(
        protected_writes=("**/.git/**",),
        protected_reads=(),
        exempt=(),
    )
    with pytest.raises((AttributeError, Exception)):
        policy.protected_writes = ("changed",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Writes — each default entry blocks when is_write=True
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
def test_default_policy_blocks_writes_to_protected_globs(path_str: str) -> None:
    assert is_protected(path_str, DEFAULT_SAFETY, is_write=True) is True


def test_default_policy_blocks_writes_to_home_rc_files() -> None:
    home = Path.home()
    for rc in (".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"):
        target = home / rc
        assert is_protected(target, DEFAULT_SAFETY, is_write=True) is True, target


# ---------------------------------------------------------------------------
# Reads — narrower list; .git/ and .aura/ reads are LEGITIMATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path_str",
    [
        "/home/alice/.ssh/id_rsa",
        "/home/alice/.ssh/known_hosts",
        "/etc/passwd",
        "/etc/nginx/nginx.conf",
    ],
)
def test_default_policy_blocks_reads_of_secret_paths(path_str: str) -> None:
    assert is_protected(path_str, DEFAULT_SAFETY, is_write=False) is True


def test_default_policy_blocks_reads_of_home_rc_files() -> None:
    home = Path.home()
    for rc in (".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"):
        target = home / rc
        assert is_protected(target, DEFAULT_SAFETY, is_write=False) is True, target


@pytest.mark.parametrize(
    "path_str",
    [
        "/some/project/.git/config",
        "/deep/nested/tree/.git/refs/heads/main",
        "/deep/nested/tree/.git/HEAD",
        "/workspace/.aura/settings.json",
        "/workspace/.aura/logs/events.jsonl",
    ],
)
def test_reads_of_git_and_aura_are_allowed(path_str: str) -> None:
    # Agent legitimately reads git metadata (git log) and aura config
    # (self-introspection). Reads list MUST NOT block these.
    assert is_protected(path_str, DEFAULT_SAFETY, is_write=False) is False


def test_ssh_blocks_both_reads_and_writes() -> None:
    target = "/home/alice/.ssh/id_rsa"
    assert is_protected(target, DEFAULT_SAFETY, is_write=True) is True
    assert is_protected(target, DEFAULT_SAFETY, is_write=False) is True


def test_etc_blocks_both_reads_and_writes() -> None:
    target = "/etc/shadow"
    assert is_protected(target, DEFAULT_SAFETY, is_write=True) is True
    assert is_protected(target, DEFAULT_SAFETY, is_write=False) is True


# ---------------------------------------------------------------------------
# Non-matching paths
# ---------------------------------------------------------------------------


def test_unprotected_path_returns_false_for_either_direction() -> None:
    target = "/Users/alice/projects/hello/src/main.py"
    assert is_protected(target, DEFAULT_SAFETY, is_write=True) is False
    assert is_protected(target, DEFAULT_SAFETY, is_write=False) is False


def test_path_that_contains_git_as_substring_but_not_component_is_not_blocked() -> None:
    target = "/repo/docs/gitignore-primer.md"
    assert is_protected(target, DEFAULT_SAFETY, is_write=True) is False
    assert is_protected(target, DEFAULT_SAFETY, is_write=False) is False


# ---------------------------------------------------------------------------
# Exempt overrides protected (both directions)
# ---------------------------------------------------------------------------


def test_exempt_entry_overrides_write_protected() -> None:
    policy = SafetyPolicy(
        protected_writes=("**/.git/**",),
        protected_reads=(),
        exempt=("**/.git/safe-config",),
    )
    assert is_protected("/repo/.git/safe-config", policy, is_write=True) is False
    assert is_protected("/repo/.git/config", policy, is_write=True) is True


def test_exempt_entry_overrides_read_protected() -> None:
    policy = SafetyPolicy(
        protected_writes=(),
        protected_reads=("**/.ssh/**",),
        exempt=("**/.ssh/config-public",),
    )
    assert is_protected("/home/a/.ssh/config-public", policy, is_write=False) is False
    assert is_protected("/home/a/.ssh/id_rsa", policy, is_write=False) is True


def test_empty_exempt_does_not_override_anything() -> None:
    policy = SafetyPolicy(
        protected_writes=("**/.git/**",),
        protected_reads=(),
        exempt=(),
    )
    assert is_protected("/repo/.git/config", policy, is_write=True) is True


def test_exempt_without_matching_protected_returns_false() -> None:
    policy = SafetyPolicy(
        protected_writes=("**/.git/**",),
        protected_reads=(),
        exempt=("/tmp/*",),
    )
    assert is_protected("/tmp/hello.txt", policy, is_write=True) is False


# ---------------------------------------------------------------------------
# Path normalization — relative paths, symlinks, tilde expansion
# ---------------------------------------------------------------------------


def test_relative_path_is_resolved_against_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src" / ".git").mkdir(parents=True)
    (tmp_path / "src" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    assert is_protected("./src/.git/HEAD", DEFAULT_SAFETY, is_write=True) is True


def test_symlink_into_protected_dir_is_blocked(tmp_path: Path) -> None:
    real_git = tmp_path / ".git"
    real_git.mkdir()
    real_head = real_git / "HEAD"
    real_head.write_text("ref: refs/heads/main\n")
    link = tmp_path / "head-link"
    try:
        os.symlink(real_head, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/fs")
    assert is_protected(link, DEFAULT_SAFETY, is_write=True) is True


def test_nonexistent_path_does_not_crash(tmp_path: Path) -> None:
    ghost = tmp_path / "does" / "not" / "exist" / ".git" / "HEAD"
    assert is_protected(ghost, DEFAULT_SAFETY, is_write=True) is True


def test_home_rc_pattern_matches_expanded_tilde_target() -> None:
    target = Path.home() / ".bashrc"
    assert is_protected(target, DEFAULT_SAFETY, is_write=True) is True
    assert is_protected(target, DEFAULT_SAFETY, is_write=False) is True


def test_tilde_in_input_path_is_expanded() -> None:
    assert is_protected("~/.bashrc", DEFAULT_SAFETY, is_write=True) is True
    assert is_protected("~/.bashrc", DEFAULT_SAFETY, is_write=False) is True


# ---------------------------------------------------------------------------
# Defensive — never crash
# ---------------------------------------------------------------------------


def test_none_input_returns_false_without_crashing() -> None:
    assert is_protected(cast(Path, None), DEFAULT_SAFETY, is_write=True) is False
    assert is_protected(cast(Path, None), DEFAULT_SAFETY, is_write=False) is False


def test_empty_string_input_returns_false_without_crashing() -> None:
    assert is_protected("", DEFAULT_SAFETY, is_write=True) is False
    assert is_protected("", DEFAULT_SAFETY, is_write=False) is False


def test_garbage_object_returns_false_without_crashing() -> None:
    assert is_protected(cast(Path, 12345), DEFAULT_SAFETY, is_write=True) is False
    assert is_protected(cast(Path, 12345), DEFAULT_SAFETY, is_write=False) is False


def test_malformed_patterns_do_not_crash_is_protected() -> None:
    policy = SafetyPolicy(
        protected_writes=("**/.git/**",),
        protected_reads=(),
        exempt=(),
    )
    assert isinstance(
        is_protected("/repo/.git/config", policy, is_write=True), bool,
    )
