"""项目记忆三层加载器 —— User / Project / Local 按规范组装。

Task 1 范围：纯发现层（3-层 walk-up）。
Task 2 范围：在 load 时展开 `@imports`（深度 5、环检测、围栏感知）。
Task 3 范围：按 resolved cwd memoize + `clear_cache`。
"""

from __future__ import annotations

from pathlib import Path

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"

_MAX_IMPORT_DEPTH = 5

# Aura 单 event-loop 运行，无并发写入；因此缓存无需加锁。
_primary_cache: dict[Path, str] = {}


def load_project_memory(cwd: Path, *, force_reload: bool = False) -> str:
    """按 User / Project(outer→inner) / Local(outer→inner) 顺序拼接项目记忆。

    文件间与层间统一以单空行分隔；缺失 / 目录占位 / 权限拒绝 —— 一律静默跳过。
    读到的文件同时展开 `@imports`（B4）。
    以 resolved cwd 为 key 进入 `_primary_cache`；`force_reload=True` 旁路缓存并覆盖。
    """
    resolved = cwd.resolve()
    if not force_reload and resolved in _primary_cache:
        return _primary_cache[resolved]

    ancestors = _ancestors_outer_to_inner(resolved)

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


def _ancestors_outer_to_inner(resolved_cwd: Path) -> list[Path]:
    # Path.parents 是 inner→outer；反转后拼上 cwd 自身，即 root-most 最先、cwd 最后
    return [*reversed(list(resolved_cwd.parents)), resolved_cwd]


def _read_raw(path: Path) -> str | None:
    # is_file() 同时挡掉 "不存在" 与 "是目录" 两种情况
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def read_with_imports(path: Path) -> str | None:
    """读单个文件并展开 `@imports`；缺失 / 目录 / 权限拒绝 → None。

    作为 project_memory 的公共 API，被 Context (B3 progressive 加载) 复用。
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

    `stripped_line` 由调用方 `rstrip()` 得到；前导空白不算 import（spec rule 11）。
    """
    if len(stripped_line) < 2 or not stripped_line.startswith("@"):
        return None
    return stripped_line[1:]


def _resolve_import(raw: str, base_dir: Path) -> Path | None:
    """按 B4 规则把 `@<raw>` 解析为 resolved Path，若为目录/不存在则返回 None。"""
    if raw.startswith("~/"):
        # 走 Path.home()，便于测试 monkeypatch；spec B4 允许 `@~/path`
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
    return resolved
