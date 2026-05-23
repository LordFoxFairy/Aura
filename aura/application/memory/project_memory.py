"""Project memory loader — User / Project walk-up / Local layers with `@imports` expansion."""

from __future__ import annotations

import subprocess
from pathlib import Path

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"

_MAX_IMPORT_DEPTH = 5

_DEFAULT_BYTE_CAP = 25_000

# Whitelist guards against pulling binary / opaque files into the prompt.
_TEXT_IMPORT_EXTS = frozenset(
    {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".sh", ".cfg", ".ini"}
)

# Single event-loop — no concurrent writes, no lock needed.
_primary_cache: dict[tuple[Path, Path | None], str] = {}


def load_project_memory(
    cwd: Path,
    *,
    force_reload: bool = False,
    auto_memory_dir: Path | None = None,
) -> str:
    """Concat User → Project(outer→inner) → Local(outer→inner) → auto-memory `MEMORY.md`."""
    resolved = cwd.resolve()
    cache_key = (resolved, auto_memory_dir.resolve() if auto_memory_dir else None)
    if not force_reload and cache_key in _primary_cache:
        return _primary_cache[cache_key]

    # Stop walking at git root when in a repo; otherwise walk to filesystem root.
    git_root = _detect_git_root(resolved)
    ancestors = _ancestors_capped(resolved, git_root)

    fragments: list[str] = []

    user_content = read_with_imports(Path.home() / _AURA_DIR / _AURA_MD)
    if user_content is not None:
        fragments.append(user_content)

    for ancestor in ancestors:
        top = read_with_imports(ancestor / _AURA_MD)
        if top is not None:
            fragments.append(top)
        nested = read_with_imports(ancestor / _AURA_DIR / _AURA_MD)
        if nested is not None:
            fragments.append(nested)

    for ancestor in ancestors:
        local = read_with_imports(ancestor / _AURA_LOCAL_MD)
        if local is not None:
            fragments.append(local)

    if auto_memory_dir is not None:
        memory_md = read_with_imports(auto_memory_dir / "MEMORY.md")
        if memory_md is not None:
            fragments.append(memory_md)

    result = "\n\n".join(fragments)
    _primary_cache[cache_key] = result
    return result


def clear_cache(cwd: Path | None = None) -> None:
    if cwd is None:
        _primary_cache.clear()
        return
    # Cache key is (resolved_cwd, auto_memory_dir | None) — drop every variant.
    target = cwd.resolve()
    for key in [k for k in _primary_cache if k[0] == target]:
        _primary_cache.pop(key, None)


def _detect_git_root(cwd: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return Path(out).resolve()
    except OSError:
        return None


def _ancestors_capped(resolved_cwd: Path, git_root: Path | None) -> list[Path]:
    """Inclusive outer→inner walk, capped at `git_root` when present, else fs root."""
    if git_root is None:
        return [*reversed(list(resolved_cwd.parents)), resolved_cwd]
    chain: list[Path] = [resolved_cwd]
    current = resolved_cwd
    while current != git_root:
        parent = current.parent
        if parent == current:
            # cwd is not under git_root — degrade to fs-root walk.
            return [*reversed(list(resolved_cwd.parents)), resolved_cwd]
        chain.append(parent)
        if parent == git_root:
            break
        current = parent
    chain.reverse()
    return chain


def _read_raw(path: Path, *, byte_cap: int = _DEFAULT_BYTE_CAP) -> str | None:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > byte_cap:
        # WARNING marker is model-facing — guides user to split the file.
        head = data[:byte_cap].decode("utf-8", errors="replace")
        return head + (
            f"\nWARNING: this file is {len(data)} bytes (limit: {byte_cap}). "
            "Keep memory files under 25 KB; split long content into separate files."
        )
    return data.decode("utf-8", errors="replace")


def read_with_imports(path: Path) -> str | None:
    """Read one file and expand `@imports` recursively; missing / dir / perm-denied → None."""
    raw = _read_raw(path)
    if raw is None:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return raw
    return _expand(raw, resolved, visited=frozenset({resolved}), depth=0)


def _expand(text: str, source: Path, *, visited: frozenset[Path], depth: int) -> str:
    """Recursive `@imports` expansion with code-fence awareness, cycle drop, depth cap."""
    out: list[str] = []
    in_fence = False
    base_dir = source.parent

    for line in text.splitlines(keepends=True):
        # Fence toggle only on column-0 triple-backtick (no leading whitespace).
        stripped_end = line.rstrip()
        if stripped_end[:3] == "```":
            in_fence = not in_fence
            out.append(line)
            continue

        if not in_fence:
            target = _parse_import(stripped_end)
            if target is not None:
                if depth + 1 >= _MAX_IMPORT_DEPTH:
                    continue
                resolved_target = _resolve_import(target, base_dir)
                if resolved_target is None or resolved_target in visited:
                    continue
                child_raw = _read_raw(resolved_target)
                if child_raw is None:
                    continue
                expanded = _expand(
                    child_raw,
                    resolved_target,
                    visited=visited | {resolved_target},
                    depth=depth + 1,
                )
                out.append(expanded)
                # Preserve line break between `@path` line and following content.
                if line.endswith(("\n", "\r")) and not expanded.endswith(("\n", "\r")):
                    out.append("\n")
                continue

        out.append(line)

    return "".join(out)


def _parse_import(stripped_line: str) -> str | None:
    # Leading whitespace doesn't count — aligns with fence detection.
    if len(stripped_line) < 2 or not stripped_line.startswith("@"):
        return None
    return stripped_line[1:]


def _resolve_import(raw: str, base_dir: Path) -> Path | None:
    if raw.startswith("~/"):
        # Path.home() (not os.path.expanduser) so tests can monkeypatch.
        candidate = Path.home() / raw[2:]
    elif raw == "~":
        candidate = Path.home()
    elif raw.startswith("/"):
        candidate = Path(raw)
    else:
        candidate = base_dir / raw
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    if resolved.suffix.lower() not in _TEXT_IMPORT_EXTS:
        try:
            from aura.core import journal

            journal.write(
                "import_non_text_skipped",
                path=str(resolved),
                suffix=resolved.suffix,
            )
        except Exception:  # noqa: BLE001  # log + swallow; logging path must never crash caller
            import logging

            logging.getLogger(__name__).warning(
                "import_non_text_skipped: %s (suffix=%r)",
                resolved,
                resolved.suffix,
            )
        return None
    return resolved
