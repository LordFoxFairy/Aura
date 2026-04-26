"""项目记忆加载器。

三层 walk-up 发现（User / Project / Local），`@imports` 预展开（深度上限
5、环检测、代码围栏感知），按 resolved cwd memoize，session 内可 clear。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"

_MAX_IMPORT_DEPTH = 5

_DEFAULT_BYTE_CAP = 25_000

# 仅这些扩展名允许通过 `@imports` 注入 — 防止误把二进制 / 不透明文件灌进 prompt
# (audit F-03-008)。
_TEXT_IMPORT_EXTS = frozenset(
    {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml", ".sh", ".cfg", ".ini"}
)

# Aura 单 event-loop 运行，无并发写入；因此缓存无需加锁。
_primary_cache: dict[Path, str] = {}


def load_project_memory(cwd: Path, *, force_reload: bool = False) -> str:
    """按 User / Project(outer→inner) / Local(outer→inner) 顺序拼接项目记忆。

    文件间与层间统一以单空行分隔；缺失 / 目录占位 / 权限拒绝 —— 一律静默跳过。
    读到的文件同时展开 `@imports`。
    以 resolved cwd 为 key 进入 `_primary_cache`；`force_reload=True` 旁路缓存并覆盖。
    """
    resolved = cwd.resolve()
    if not force_reload and resolved in _primary_cache:
        return _primary_cache[resolved]

    # F-03-006: stop walking at the git root if `cwd` is in a repo. Falls
    # back to filesystem root when `git` is missing or `cwd` isn't tracked,
    # preserving the legacy behaviour for non-git trees.
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

    result = "\n\n".join(fragments)
    _primary_cache[resolved] = result
    return result


def clear_cache(cwd: Path | None = None) -> None:
    """清 `_primary_cache`：无参清空全部；有参清指定 resolved cwd（不存在则静默）。"""
    if cwd is None:
        _primary_cache.clear()
        return
    _primary_cache.pop(cwd.resolve(), None)


def _detect_git_root(cwd: Path) -> Path | None:
    """F-03-006 helper: probe `git rev-parse --show-toplevel`. Tolerant of
    missing git (`FileNotFoundError`), non-repos (non-zero rc), and slow
    filesystems (2s timeout)."""
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
    """Inclusive walk from outermost-allowed ancestor down to `resolved_cwd`.

    When `git_root` is provided, the walk stops at it (inclusive); otherwise
    it falls back to filesystem root, mirroring the legacy behaviour.
    """
    if git_root is None:
        return [*reversed(list(resolved_cwd.parents)), resolved_cwd]
    chain: list[Path] = [resolved_cwd]
    current = resolved_cwd
    while current != git_root:
        parent = current.parent
        if parent == current:
            # `cwd` 不在 `git_root` 之下（罕见的边界）—— 退化到 fs-root walk
            return [*reversed(list(resolved_cwd.parents)), resolved_cwd]
        chain.append(parent)
        if parent == git_root:
            break
        current = parent
    chain.reverse()
    return chain


def _read_raw(path: Path, *, byte_cap: int = _DEFAULT_BYTE_CAP) -> str | None:
    # is_file() 同时挡掉 "不存在" 与 "是目录" 两种情况
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > byte_cap:
        # F-03-007: 对超出 byte_cap 的 memory 文件追加 WARNING marker；
        # 截断后追加文本以引导用户拆文件。
        head = data[:byte_cap].decode("utf-8", errors="replace")
        return head + (
            f"\nWARNING: this file is {len(data)} bytes (limit: {byte_cap}). "
            "Keep memory files under 25 KB; split long content into separate files."
        )
    return data.decode("utf-8", errors="replace")


def read_with_imports(path: Path) -> str | None:
    """读单个文件并展开 `@imports`；缺失 / 目录 / 权限拒绝 → None。

    公共 API：供 Context 的子目录按需加载路径复用同一套 `@imports` 解析。
    """
    raw = _read_raw(path)
    if raw is None:
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return raw
    return _expand(raw, resolved, visited=frozenset({resolved}), depth=0)


def _expand(text: str, source: Path, *, visited: frozenset[Path], depth: int) -> str:
    """展开 `text` 中的 `@imports`：
    - `source` 是该文本所在已解析文件路径（其 parent 为相对路径基准）。
    - `visited` 沿递归链传递，用于环检测（drop 即可，不抛）。
    - `depth` 为当前文件在递归链中的深度（根文件=0）；子文件深度 >= 5 时丢弃。
    """
    out: list[str] = []
    in_fence = False
    base_dir = source.parent

    for line in text.splitlines(keepends=True):
        # 判断围栏切换：首三字符为三反引号（column-0，无前导空白）
        stripped_end = line.rstrip()
        if stripped_end[:3] == "```":
            in_fence = not in_fence
            out.append(line)
            continue

        if not in_fence:
            target = _parse_import(stripped_end)
            if target is not None:
                if depth + 1 >= _MAX_IMPORT_DEPTH:
                    continue  # 超深度静默丢弃
                resolved_target = _resolve_import(target, base_dir)
                if resolved_target is None or resolved_target in visited:
                    continue  # 缺失/目录/权限/环 —— 静默丢弃
                child_raw = _read_raw(resolved_target)
                if child_raw is None:
                    continue
                expanded = _expand(
                    child_raw,
                    resolved_target,
                    visited=visited | {resolved_target},
                    depth=depth + 1,
                )
                # 被替换的 `@path` 行自身（含换行）完全消失，
                # 子内容按其原样注入；若子内容不以换行结尾，保留原状（下一行紧跟）。
                out.append(expanded)
                # 若原 `@path` 行带换行而 expanded 不含末尾换行，补一个以分隔后续行。
                if line.endswith(("\n", "\r")) and not expanded.endswith(("\n", "\r")):
                    out.append("\n")
                continue

        out.append(line)

    return "".join(out)


def _parse_import(stripped_line: str) -> str | None:
    """若一行形如 `^@<path>$`（rstrip 后），返回路径串；否则 None。

    `stripped_line` 由调用方 `rstrip()` 得到；前导空白不识别为 import —— 与代码
    围栏检测对齐，避免缩进内的 `@` 被误吃。
    """
    if len(stripped_line) < 2 or not stripped_line.startswith("@"):
        return None
    return stripped_line[1:]


def _resolve_import(raw: str, base_dir: Path) -> Path | None:
    """把 `@<raw>` 解析为 resolved Path；若目标为目录/不存在/扩展名不在白名单则 None。

    F-03-008: 仅允许 `_TEXT_IMPORT_EXTS` 中的扩展名穿过 `@imports`，避免随手
    `@./binary.exe` 把不透明字节灌进 prompt。被拒绝的尝试发一条 journal 事件。
    """
    if raw.startswith("~/"):
        # 走 Path.home() 而非 os.path.expanduser —— 后者读 $HOME env，
        # 测试无法通过 monkeypatch Path.home 覆盖。
        candidate = Path.home() / raw[2:]
    elif raw == "~":
        candidate = Path.home()
    elif raw.startswith("/"):
        candidate = Path(raw)
    else:
        # `./x` 与 `x` 均相对于 importing file 的 parent
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
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "import_non_text_skipped: %s (suffix=%r)",
                resolved,
                resolved.suffix,
            )
        return None
    return resolved
