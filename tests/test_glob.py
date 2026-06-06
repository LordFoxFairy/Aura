"""Tests for aura.tools.glob."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from aura.domain.tool import ToolError
from aura.domain.tool_meta_access import meta_dict
from aura.tools.glob import GlobParams, glob

_GLOB_MOD: ModuleType = importlib.import_module("aura.tools.glob")


async def test_glob_finds_py_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["count"] == 2
    assert all(f.endswith(".py") for f in out["files"])


async def test_glob_recursive_double_star(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "top.py").write_text("", encoding="utf-8")
    (sub / "nested.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "**/*.py", "path": str(tmp_path)})
    assert out["count"] == 2


async def test_glob_respects_max_results(tmp_path: Path) -> None:
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke(
        {"pattern": "*.py", "path": str(tmp_path), "max_results": 3}
    )
    assert out["count"] == 3
    assert out["truncated"] is True


async def test_glob_missing_path_returns_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="not found"):
        await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path / "nope")})


async def test_glob_empty_result_ok(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["count"] == 0
    assert out["files"] == []
    assert out["truncated"] is False


async def test_glob_returns_relative_paths(tmp_path: Path) -> None:
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "**/*.py", "path": str(tmp_path)})
    assert len(out["files"]) == 1
    assert not out["files"][0].startswith("/")


def test_glob_capability_flags() -> None:
    meta = meta_dict(glob)
    assert meta.get("is_read_only") is True
    assert meta.get("is_concurrency_safe") is True
    assert meta.get("is_destructive") is False


def test_glob_metadata_includes_matcher_and_preview() -> None:
    from aura.tools.glob import glob

    meta = meta_dict(glob)
    assert meta.get("rule_matcher") is not None
    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"pattern": "**/*.py"}) == "pattern: **/*.py  @ ."


def _set_mtime(p: Path, mtime: float) -> None:
    os.utime(p, (mtime, mtime))


async def test_default_sort_is_mtime_newest_first(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for f in (a, b, c):
        f.write_text("", encoding="utf-8")
    _set_mtime(c, 1_000.0)
    _set_mtime(a, 2_000.0)
    _set_mtime(b, 3_000.0)
    out = await glob.ainvoke({"pattern": "*.txt", "path": str(tmp_path)})
    assert out["files"] == ["b.txt", "a.txt", "c.txt"]


async def test_alphabetical_sort_opt_in(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for f in (a, b, c):
        f.write_text("", encoding="utf-8")
    _set_mtime(c, 1_000.0)
    _set_mtime(a, 2_000.0)
    _set_mtime(b, 3_000.0)
    out = await glob.ainvoke(
        {"pattern": "*.txt", "path": str(tmp_path), "sort": "alphabetical"}
    )
    assert out["files"] == ["a.txt", "b.txt", "c.txt"]


async def test_mtime_sort_stable_when_equal_mtimes(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    for f in (a, b):
        f.write_text("", encoding="utf-8")
    _set_mtime(a, 5_000.0)
    _set_mtime(b, 5_000.0)
    out = await glob.ainvoke({"pattern": "*.txt", "path": str(tmp_path)})
    assert out["count"] == 2
    assert set(out["files"]) == {"a.txt", "b.txt"}


async def test_mtime_sort_skips_vanished_files_safely(tmp_path: Path) -> None:
    # Race-safe emulation: create files, then make stat unreliable on one by
    # deleting it before the tool stats. The tool enumerates via Path.glob()
    # which captures the name list; stat() is called per file during sort.
    # We can't reliably race the tool, so instead we rely on the fact that
    # OSError on stat falls back to mtime=0.0. Instead verify it doesn't crash
    # when the dir contains files and all stats succeed. Coverage for the
    # fallback path is left to the implementation's try/except (exercised via
    # manual inspection). We do assert no crash and sorted output is sane.
    for name in ("x.txt", "y.txt", "z.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "*.txt", "path": str(tmp_path)})
    assert out["count"] == 3
    assert set(out["files"]) == {"x.txt", "y.txt", "z.txt"}


def test_invalid_sort_value_rejected() -> None:
    from aura.tools.glob import GlobParams

    with pytest.raises(ValidationError):
        GlobParams.model_validate({"pattern": "*.py", "sort": "lastmod"})


def _git_init_repo(root: Path) -> None:
    subprocess.run(
        ["git", "init", "-q"], cwd=root, check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "t@t"], check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "t"], check=True,
        capture_output=True,
    )


def _git_add_commit(root: Path) -> None:
    subprocess.run(
        ["git", "-C", str(root), "add", "-A"], check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "init"], check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
async def test_glob_in_git_repo_excludes_gitignored_file(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("", encoding="utf-8")
    (tmp_path / "secret.py").write_text("", encoding="utf-8")
    _git_add_commit(tmp_path)

    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert "ok.py" in out["files"]
    assert "secret.py" not in out["files"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
async def test_glob_in_git_repo_excludes_untracked_file(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    (tmp_path / "tracked.py").write_text("", encoding="utf-8")
    _git_add_commit(tmp_path)
    # New file added after initial commit and never `git add`-ed.
    (tmp_path / "untracked.py").write_text("", encoding="utf-8")

    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert "tracked.py" in out["files"]
    assert "untracked.py" not in out["files"]


async def test_glob_in_non_git_dir_includes_all_files(tmp_path: Path) -> None:
    # Same names as the gitignore test, but no .git/ → ignored file IS included.
    (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("", encoding="utf-8")
    (tmp_path / "secret.py").write_text("", encoding="utf-8")

    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert "ok.py" in out["files"]
    assert "secret.py" in out["files"]


async def test_truncation_preserves_sort_order(tmp_path: Path) -> None:
    files = [tmp_path / f"f{i}.txt" for i in range(10)]
    for f in files:
        f.write_text("", encoding="utf-8")
    # Assign distinct mtimes: f0 oldest, f9 newest.
    for i, f in enumerate(files):
        _set_mtime(f, 1_000.0 + i)
    out = await glob.ainvoke(
        {"pattern": "*.txt", "path": str(tmp_path), "max_results": 3}
    )
    assert out["count"] == 3
    assert out["truncated"] is True
    assert out["files"] == ["f9.txt", "f8.txt", "f7.txt"]


async def test_glob_path_is_a_file_rejected(tmp_path: Path) -> None:
    """A file path (not a dir) must be refused so callers can't smuggle reads."""
    target = tmp_path / "a_file.py"
    target.write_text("", encoding="utf-8")
    with pytest.raises(ToolError, match="not a directory"):
        await glob.ainvoke({"pattern": "*.py", "path": str(target)})


async def test_git_absent_keeps_all_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without git on PATH the repo filter is skipped, not silently empty."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "ok.py").write_text("", encoding="utf-8")
    (tmp_path / "more.py").write_text("", encoding="utf-8")
    fake_shutil = SimpleNamespace(which=lambda name: None)
    monkeypatch.setattr(_GLOB_MOD, "shutil", fake_shutil)
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert set(out["files"]) == {"ok.py", "more.py"}


async def test_git_ls_files_nonzero_falls_back_to_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing `git ls-files` must not hide files; degrade to no filtering."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "ok.py").write_text("", encoding="utf-8")
    (tmp_path / "more.py").write_text("", encoding="utf-8")

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=128, stdout=b"")

    monkeypatch.setattr(_GLOB_MOD, "shutil", SimpleNamespace(which=lambda n: "git"))
    monkeypatch.setattr(
        _GLOB_MOD,
        "subprocess",
        SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert set(out["files"]) == {"ok.py", "more.py"}


async def test_git_ls_files_timeout_falls_back_to_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung `git ls-files` must time out and degrade, never block forever."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "ok.py").write_text("", encoding="utf-8")

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd="git", timeout=10)

    monkeypatch.setattr(_GLOB_MOD, "shutil", SimpleNamespace(which=lambda n: "git"))
    monkeypatch.setattr(
        _GLOB_MOD,
        "subprocess",
        SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["files"] == ["ok.py"]


async def test_git_ls_files_oserror_falls_back_to_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If spawning git raises OSError the tool still returns matches."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "ok.py").write_text("", encoding="utf-8")

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        raise OSError("cannot exec git")

    monkeypatch.setattr(_GLOB_MOD, "shutil", SimpleNamespace(which=lambda n: "git"))
    monkeypatch.setattr(
        _GLOB_MOD,
        "subprocess",
        SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["files"] == ["ok.py"]


async def test_git_tracked_set_filters_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NUL-delimited tracked set must exclude files git does not list."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "ok.py").write_text("", encoding="utf-8")
    (tmp_path / "untracked.py").write_text("", encoding="utf-8")
    tracked_payload = b"ok.py\x00"

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=tracked_payload)

    monkeypatch.setattr(_GLOB_MOD, "shutil", SimpleNamespace(which=lambda n: "git"))
    monkeypatch.setattr(
        _GLOB_MOD,
        "subprocess",
        SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["files"] == ["ok.py"]


async def test_git_tracked_set_empty_excludes_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty tracked set (fresh repo, nothing added) yields zero matches."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "ok.py").write_text("", encoding="utf-8")

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(_GLOB_MOD, "shutil", SimpleNamespace(which=lambda n: "git"))
    monkeypatch.setattr(
        _GLOB_MOD,
        "subprocess",
        SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert out["files"] == []
    assert out["count"] == 0


def test_safe_mtime_returns_zero_when_stat_fails() -> None:
    """A file that vanishes before stat must sort as oldest, not crash the run."""
    assert _GLOB_MOD._safe_mtime(Path("/nonexistent/zzz/qqq")) == 0.0


async def test_pathlib_glob_matches_dotfiles(tmp_path: Path) -> None:
    """Pathlib '*' matches dotfiles (unlike shell), so hidden files surface."""
    (tmp_path / ".hidden.py").write_text("", encoding="utf-8")
    (tmp_path / "visible.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": "*.py", "path": str(tmp_path)})
    assert set(out["files"]) == {".hidden.py", "visible.py"}


async def test_dot_pattern_isolates_hidden_file(tmp_path: Path) -> None:
    """An explicit leading-dot pattern targets only the hidden file."""
    (tmp_path / ".hidden.py").write_text("", encoding="utf-8")
    (tmp_path / "visible.py").write_text("", encoding="utf-8")
    out = await glob.ainvoke({"pattern": ".*.py", "path": str(tmp_path)})
    assert out["files"] == [".hidden.py"]


async def test_count_equal_to_cap_is_not_truncated(tmp_path: Path) -> None:
    """Off-by-one guard: exactly max_results matches must report truncated=False."""
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("", encoding="utf-8")
    out = await glob.ainvoke(
        {"pattern": "*.txt", "path": str(tmp_path), "max_results": 3}
    )
    assert out["count"] == 3
    assert out["truncated"] is False


async def test_empty_pattern_surfaces_value_error(tmp_path: Path) -> None:
    """An empty pattern is a caller error; pathlib's ValueError must not be eaten."""
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="pattern"):
        await glob.ainvoke({"pattern": "", "path": str(tmp_path)})


async def test_glob_is_idempotent_across_repeated_calls(tmp_path: Path) -> None:
    """A read-only search must return identical results when called twice."""
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    first = await glob.ainvoke(
        {"pattern": "*.py", "path": str(tmp_path), "sort": "alphabetical"}
    )
    second = await glob.ainvoke(
        {"pattern": "*.py", "path": str(tmp_path), "sort": "alphabetical"}
    )
    assert first == second


def test_max_results_below_floor_rejected() -> None:
    """max_results has ge=1; zero/negative caps are nonsensical and refused."""
    with pytest.raises(ValidationError):
        GlobParams.model_validate({"pattern": "*.py", "max_results": 0})


def test_max_results_above_ceiling_rejected() -> None:
    """max_results has le=10_000; an unbounded cap would defeat truncation."""
    with pytest.raises(ValidationError):
        GlobParams.model_validate({"pattern": "*.py", "max_results": 10_001})


def test_missing_pattern_rejected() -> None:
    """pattern is required with no default; omitting it must fail validation."""
    with pytest.raises(ValidationError):
        GlobParams.model_validate({"path": "."})


def test_glob_params_defaults() -> None:
    """Defaults match the documented contract: cwd, 500-cap, mtime ordering."""
    params = GlobParams.model_validate({"pattern": "**/*.py"})
    assert params.path == "."
    assert params.max_results == 500
    assert params.sort == "mtime"
