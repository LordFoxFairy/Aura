"""`.aura/rules/*.md` 的发现、frontmatter 解析与 glob 匹配。

两层扫描：user (`~/.aura/rules/**/*.md`) 与 project (`<cwd>/.aura/rules/**/*.md`，
仅 cwd 自身，不向祖先 walk-up)。按 YAML frontmatter 的 `paths` 字段把结果切成
`unconditional`（无 `paths`，随 primary memory eager 注入）与 `conditional`
（有 `paths`，由 `match()` 在 tool 触达路径时按需挑选）。

坏掉的单条规则（非 UTF-8 / YAML 错误 / glob 编译失败 / 权限拒绝）一律静默跳过，
避免一条规则把整个 session 的 memory 层打挂。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pathspec
import yaml

_AURA_DIR = ".aura"
_RULES_DIR = "rules"
_MD_SUFFIX = ".md"

_MAX_LINES = 200
_DEFAULT_BYTE_CAP = 25_000


def _truncation_warning(actual_bytes: int, limit: int) -> str:
    return (
        f"\nWARNING: this file is {actual_bytes} bytes (limit: {limit}). "
        "Keep memory files under 25 KB; split long content into separate files."
    )


@dataclass(frozen=True)
class Rule:
    """单条规则文件的元数据 + 截断后的正文。"""

    source_path: Path  # .aura/rules/<...>/x.md 的 resolved 绝对路径
    base_dir: Path  # 持有 .aura/rules/ 的父目录（user: ~；project: cwd）
    globs: tuple[str, ...]  # frontmatter `paths:` 解析出的 glob；空元组 = unconditional
    content: str  # body（去掉 frontmatter 后），已按 200 行/4KB 截断


@dataclass
class RulesBundle:
    """一次 load 的结果：eager 桶随 primary memory 注入，conditional 桶按 path 触发。"""

    unconditional: list[Rule] = field(default_factory=list)
    conditional: list[Rule] = field(default_factory=list)


# Aura 单 event-loop，无并发写；缓存无需加锁。
_rules_cache: dict[Path, RulesBundle] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_rules(cwd: Path, *, force_reload: bool = False) -> RulesBundle:
    """扫描 user + project 两层 `.aura/rules/**/*.md`，按 resolved cwd memoize。

    `force_reload=True` 旁路缓存并覆盖。
    """
    resolved_cwd = cwd.resolve()
    if not force_reload and resolved_cwd in _rules_cache:
        return _rules_cache[resolved_cwd]

    bundle = RulesBundle()

    home = Path.home()
    _scan_layer(home / _AURA_DIR / _RULES_DIR, base_dir=home, bundle=bundle)

    # project 层仅看 cwd 自身的 `.aura/rules/`，不向祖先 walk-up：规则是 "项目级"
    # 而非 "路径级" 的配置，walk-up 会让祖先项目的规则泄漏进子项目。
    _scan_layer(
        resolved_cwd / _AURA_DIR / _RULES_DIR, base_dir=resolved_cwd, bundle=bundle
    )

    _rules_cache[resolved_cwd] = bundle
    return bundle


def clear_cache(cwd: Path | None = None) -> None:
    """无参清空全部；有参清除指定 resolved cwd 条目（不存在则静默）。"""
    if cwd is None:
        _rules_cache.clear()
        return
    _rules_cache.pop(cwd.resolve(), None)


def match(bundle: RulesBundle, path: Path) -> list[Rule]:
    """返回 `bundle.conditional` 中任意 glob 命中 `path` 的规则集合。

    - 若 `path` 在 `rule.base_dir` 之下，则以相对路径字符串匹配；否则用绝对路径字符串。
    - 去重按 `source_path`，按 `source_path` 排序。
    - 单条规则的 globs 无法编译 → 静默跳过（不计入结果）。
    """
    try:
        resolved_path = path.resolve()
    except OSError:
        resolved_path = path

    seen: set[Path] = set()
    matched: list[Rule] = []
    for rule in bundle.conditional:
        if rule.source_path in seen:
            continue
        if _rule_matches_path(rule, resolved_path):
            matched.append(rule)
            seen.add(rule.source_path)

    matched.sort(key=lambda r: r.source_path)
    return matched


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _scan_layer(rules_root: Path, *, base_dir: Path, bundle: RulesBundle) -> None:
    """递归扫描 `rules_root` 下所有 `*.md`，把 Rule 挂入 bundle。

    整个 layer 若不存在 / 非目录 / 权限拒绝 —— 整体静默跳过。
    """
    if not rules_root.is_dir():
        return
    try:
        md_files = sorted(rules_root.rglob(f"*{_MD_SUFFIX}"))
    except OSError:
        return

    for md_path in md_files:
        # rglob 也可能返回目录名为 "x.md" 的目录，保险再判一次
        if not md_path.is_file():
            continue
        rule = _build_rule(md_path, base_dir=base_dir)
        if rule is None:
            continue
        if rule.globs:
            bundle.conditional.append(rule)
        else:
            bundle.unconditional.append(rule)


def _build_rule(md_path: Path, *, base_dir: Path) -> Rule | None:
    """读取单个 .md 文件；返回 Rule 或 None（后者表示静默跳过）。"""
    raw = _read_text(md_path)
    if raw is None:
        return None

    frontmatter_text, body = _split_frontmatter(raw)

    if frontmatter_text is None:
        # 无 frontmatter → unconditional（与 AURA.md 同级 eager 注入）
        globs: tuple[str, ...] = ()
    else:
        try:
            parsed = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as exc:
            from aura.core import journal

            journal.write(
                "rule_yaml_parse_failed", path=str(md_path), error=str(exc)
            )
            return None  # 恶意 YAML → 静默跳过整条规则
        globs_or_skip = _extract_globs(parsed)
        if globs_or_skip is _SKIP:
            from aura.core import journal

            actual_type = type(parsed["paths"]).__name__ if isinstance(
                parsed, dict
            ) and "paths" in parsed else type(parsed).__name__
            journal.write(
                "rule_paths_invalid_type",
                path=str(md_path),
                actual_type=actual_type,
            )
            return None  # `paths` 是未知类型 → 跳过
        globs = globs_or_skip  # type: ignore[assignment]

    try:
        source = md_path.resolve()
        base = base_dir.resolve()
    except OSError:
        return None

    return Rule(
        source_path=source,
        base_dir=base,
        globs=globs,
        content=_truncate(body),
    )


def _read_text(path: Path) -> str | None:
    """读文件字节并以 errors='replace' 解码；缺失/目录/perm denied → None。"""
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def _split_frontmatter(raw: str) -> tuple[str | None, str]:
    """切出 YAML frontmatter 与其后的 body。

    frontmatter 规则：
      - 文件首行必须是 `---`（允许末尾空白/CRLF）。
      - 下一个独占 `---`（或 `...`）作为闭合；之后到 EOF 是 body。
      - 若闭合行不存在 → 视为"无 frontmatter"，整份作为 body 返回。

    返回 (frontmatter_text_or_None, body)。
    """
    lines = raw.splitlines(keepends=True)
    if not lines:
        return None, raw

    first = lines[0].rstrip("\r\n").rstrip()
    if first != "---":
        return None, raw

    for idx in range(1, len(lines)):
        stripped = lines[idx].rstrip("\r\n").rstrip()
        if stripped in {"---", "..."}:
            frontmatter = "".join(lines[1:idx])
            body = "".join(lines[idx + 1 :])
            return frontmatter, body

    # 无闭合 → 视为无 frontmatter
    return None, raw


# sentinel 区分 "paths 字段缺失（→ unconditional）" 与 "paths 字段类型未知（→ skip）"
_SKIP = object()


def _extract_globs(parsed: Any) -> tuple[str, ...] | object:
    """从解析后的 frontmatter dict 中取 `paths` 字段并归一化为 globs 元组。

    - parsed 不是 dict → 视作无 frontmatter dict → 空元组（unconditional）。
    - `paths` 缺失 → 空元组（unconditional）。
    - str（允许逗号分隔、各段 strip）→ 过滤空串后的元组。
    - list → 逐项转 str 的元组。
    - 其它类型 → `_SKIP`。
    - 归一化后所有 entries ∈ {"**", "**/*"} → 空元组（unconditional carve-out, F-03-005）。
    """
    if not isinstance(parsed, dict):
        return ()
    if "paths" not in parsed:
        return ()
    value = parsed["paths"]
    if isinstance(value, str):
        parts = tuple(p.strip() for p in value.split(",") if p.strip())
        return _normalize_universal(parts)
    if isinstance(value, list):
        parts = tuple(str(item) for item in value)
        return _normalize_universal(parts)
    return _SKIP


_UNIVERSAL_GLOBS = frozenset({"**", "**/*"})


def _normalize_universal(parts: tuple[str, ...]) -> tuple[str, ...]:
    if parts and all(p in _UNIVERSAL_GLOBS for p in parts):
        return ()
    return parts


def _truncate(body: str, *, byte_cap: int = _DEFAULT_BYTE_CAP) -> str:
    """200 行 或 `byte_cap` 字节（首个命中）即截断，尾部追加 WARNING marker。

    `byte_cap` 取代旧的硬编码 4096：rules 走默认 25_000，调用方可显式收紧。
    Marker 暴露 actual/limit 给模型，引导用户把超长 memory 拆文件 (F-03-007)。
    """
    original_bytes = len(body.encode("utf-8"))
    truncated = False
    limit_hit = byte_cap

    lines = body.splitlines(keepends=True)
    if len(lines) > _MAX_LINES:
        lines = lines[:_MAX_LINES]
        truncated = True

    trimmed = "".join(lines)
    encoded = trimmed.encode("utf-8")
    if len(encoded) > byte_cap:
        # 按字节截断后，末尾可能砍断一个多字节字符；errors='replace' 维持字符串完整
        trimmed = encoded[:byte_cap].decode("utf-8", errors="replace")
        truncated = True

    if truncated:
        return trimmed + _truncation_warning(original_bytes, limit_hit)
    return trimmed


def _rule_matches_path(rule: Rule, resolved_path: Path) -> bool:
    """该规则的任一 glob 命中 `resolved_path` 即 True；编译失败 → False。"""
    match_target = _relative_or_absolute(resolved_path, rule.base_dir)
    for glob in rule.globs:
        # pathspec 在编译/匹配阶段对各种畸形 glob 抛出多种异常类型
        # （`GitIgnorePatternError`、`re.error` 等）；全部静默跳过即可。
        try:
            spec = pathspec.PathSpec.from_lines("gitignore", [glob])
            if spec.match_file(match_target):
                return True
        except Exception as exc:  # noqa: BLE001
            from aura.core import journal

            journal.write(
                "rule_glob_compile_failed",
                path=str(rule.source_path),
                glob=glob,
                error=str(exc),
            )
            continue
    return False


def _relative_or_absolute(path: Path, base_dir: Path) -> str:
    """若 `path` 在 `base_dir` 之下，返回相对路径字符串；否则返回绝对路径字符串。

    `base_dir` 已是 `rule.base_dir`（已 resolved）。
    """
    try:
        rel = path.relative_to(base_dir)
        return rel.as_posix()
    except ValueError:
        return path.as_posix()
