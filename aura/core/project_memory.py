"""项目记忆三层加载器 —— User / Project / Local 按规范组装。

Task 1 范围：纯发现层（3-层 walk-up），无 @imports、无 memoize。
@imports 于 Task 2 接入，memoize + clear_cache 于 Task 3 接入。
"""

from __future__ import annotations

from pathlib import Path

_AURA_MD = "AURA.md"
_AURA_DIR = ".aura"
_AURA_LOCAL_MD = "AURA.local.md"


def load_project_memory(cwd: Path, *, force_reload: bool = False) -> str:
    """按 User / Project(outer→inner) / Local(outer→inner) 顺序拼接项目记忆。

    文件间与层间统一以单空行分隔；缺失 / 目录占位 / 权限拒绝 —— 一律静默跳过。
    `force_reload` 在 Task 3 才起作用，这里保留签名避免后续改动。
    """
    del force_reload  # Task 3 wire-in
    resolved = cwd.resolve()
    ancestors = _ancestors_outer_to_inner(resolved)

    fragments: list[str] = []

    # User 层：~/.aura/AURA.md（单文件，不 walk-up）
    user_content = _read_if_file(Path.home() / _AURA_DIR / _AURA_MD)
    if user_content is not None:
        fragments.append(user_content)

    # Project 层：每个祖先目录的 AURA.md，然后 .aura/AURA.md
    for ancestor in ancestors:
        top = _read_if_file(ancestor / _AURA_MD)
        if top is not None:
            fragments.append(top)
        nested = _read_if_file(ancestor / _AURA_DIR / _AURA_MD)
        if nested is not None:
            fragments.append(nested)

    # Local 层：每个祖先目录的 AURA.local.md
    for ancestor in ancestors:
        local = _read_if_file(ancestor / _AURA_LOCAL_MD)
        if local is not None:
            fragments.append(local)

    return "\n\n".join(fragments)


def _ancestors_outer_to_inner(resolved_cwd: Path) -> list[Path]:
    # Path.parents 是 inner→outer；反转后拼上 cwd 自身，即 root-most 最先、cwd 最后
    return [*reversed(list(resolved_cwd.parents)), resolved_cwd]


def _read_if_file(path: Path) -> str | None:
    # is_file() 同时挡掉 "不存在" 与 "是目录" 两种情况
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        # 权限拒绝等底层 IO 错误一律静默
        return None
    return data.decode("utf-8", errors="replace")
