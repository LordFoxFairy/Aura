"""F-03-004 — auto-memory pipeline.

The model owns memory writes through the existing ``write_file`` /
``read_file`` tools. Aura's role is:

1. expose a per-project memory directory via ``Storage.memory_dir(cwd=...)``
2. tell the model the convention via the system-prompt's ``# auto memory``
   section (rendered when ``auto_memory_dir`` is supplied)
3. load ``MEMORY.md`` from that directory as part of project memory so
   future sessions see what the model previously saved

These tests pin all three contracts.
"""

from __future__ import annotations

from pathlib import Path

from aura.core.memory.project_memory import clear_cache, load_project_memory
from aura.core.memory.system_prompt import build_system_prompt
from aura.core.persistence.storage import SessionStorage


def test_memory_dir_resolves_under_per_project_layout(tmp_path: Path) -> None:
    """``Storage.memory_dir`` lives next to ``<session-id>.jsonl``."""
    storage = SessionStorage(tmp_path / "sessions.db")
    project = tmp_path / "fake-project"
    project.mkdir()
    mem = storage.memory_dir(cwd=project)
    # Sibling of the session dir; under <projects>/<encoded-cwd>/memory.
    assert mem.name == "memory"
    assert mem.parent == storage._project_dir(project)
    # No side effects — we don't materialize the directory.
    assert not mem.exists()


def test_load_project_memory_includes_memory_md(tmp_path: Path) -> None:
    """When MEMORY.md exists in the auto-memory dir, it joins the eager chain."""
    storage = SessionStorage(tmp_path / "sessions.db")
    project = tmp_path / "proj"
    project.mkdir()
    mem_dir = storage.memory_dir(cwd=project)
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text(
        "# Memory Index\n\n- [Bash style](bash_style.md) — user prefers fish\n",
        encoding="utf-8",
    )

    clear_cache(project)
    content = load_project_memory(project, auto_memory_dir=mem_dir)
    assert "Memory Index" in content
    assert "bash_style.md" in content


def test_load_project_memory_handles_missing_memory_md(tmp_path: Path) -> None:
    """No MEMORY.md → loader silently degrades to non-auto-memory output."""
    storage = SessionStorage(tmp_path / "sessions.db")
    project = tmp_path / "proj"
    project.mkdir()
    mem_dir = storage.memory_dir(cwd=project)  # does NOT exist on disk

    clear_cache(project)
    content = load_project_memory(project, auto_memory_dir=mem_dir)
    # Empty (no AURA.md, no MEMORY.md) — no exception.
    assert content == ""


def test_load_project_memory_cache_keys_on_auto_memory_dir(
    tmp_path: Path,
) -> None:
    """Same cwd + different auto_memory_dir → distinct cache entries."""
    # Two storages under different roots so each owns its own
    # ``projects/<encoded>/memory/`` subtree.
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    root_a.mkdir()
    root_b.mkdir()
    storage_a = SessionStorage(root_a / "sessions.db")
    storage_b = SessionStorage(root_b / "sessions.db")
    project = tmp_path / "proj"
    project.mkdir()

    mem_a = storage_a.memory_dir(cwd=project)
    mem_b = storage_b.memory_dir(cwd=project)
    mem_a.mkdir(parents=True)
    mem_b.mkdir(parents=True)
    (mem_a / "MEMORY.md").write_text("FROM-A", encoding="utf-8")
    (mem_b / "MEMORY.md").write_text("FROM-B", encoding="utf-8")

    clear_cache(project)
    content_a = load_project_memory(project, auto_memory_dir=mem_a)
    content_b = load_project_memory(project, auto_memory_dir=mem_b)
    assert "FROM-A" in content_a
    assert "FROM-B" in content_b


def test_clear_cache_drops_all_auto_memory_variants(tmp_path: Path) -> None:
    """``clear_cache(cwd)`` must drop every (cwd, dir) entry, not just the
    legacy single-key form.
    """
    storage = SessionStorage(tmp_path / "sessions.db")
    project = tmp_path / "proj"
    project.mkdir()
    mem_dir = storage.memory_dir(cwd=project)
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("V1", encoding="utf-8")

    # Prime cache with auto_memory_dir.
    clear_cache(project)
    first = load_project_memory(project, auto_memory_dir=mem_dir)
    assert "V1" in first

    # Mutate disk state.
    (mem_dir / "MEMORY.md").write_text("V2", encoding="utf-8")
    # Cache still serves V1.
    cached = load_project_memory(project, auto_memory_dir=mem_dir)
    assert "V1" in cached

    # clear_cache(project) must drop the (project, mem_dir) entry too —
    # not just the legacy (project, None) form.
    clear_cache(project)
    fresh = load_project_memory(project, auto_memory_dir=mem_dir)
    assert "V2" in fresh


def test_system_prompt_omits_memory_section_when_dir_is_none(
    tmp_path: Path,
) -> None:
    """Backwards compat: legacy callers without ``auto_memory_dir`` see
    no auto-memory section.
    """
    prompt = build_system_prompt(cwd=tmp_path)
    assert "# auto memory" not in prompt


def test_system_prompt_includes_memory_section_when_dir_supplied(
    tmp_path: Path,
) -> None:
    """When wired with a memory dir, the prompt explains the convention.

    The exact dir path lands in the prompt so the model knows where to
    write — claude-code parity (its own auto-memory section names the
    target directory verbatim).
    """
    mem = tmp_path / "memory"
    prompt = build_system_prompt(cwd=tmp_path, auto_memory_dir=mem)
    assert "# auto memory" in prompt
    assert str(mem) in prompt
    # All four memory types are documented.
    assert "**user**" in prompt
    assert "**feedback**" in prompt
    assert "**project**" in prompt
    assert "**reference**" in prompt
    # The "what NOT to save" guidance is present.
    assert "What NOT to save" in prompt
    # MEMORY.md as the index is documented.
    assert "MEMORY.md" in prompt
