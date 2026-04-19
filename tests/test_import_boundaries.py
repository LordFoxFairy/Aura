"""Static enforcement of Aura's layer import rules."""

from __future__ import annotations

import ast
from pathlib import Path

_AURA_ROOT = Path(__file__).resolve().parent.parent / "aura"


def _module_imports(py_file: Path) -> set[str]:
    """Return the top-level module names imported by *py_file*."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _python_files(subdir: str) -> list[Path]:
    root = _AURA_ROOT / subdir
    return sorted(root.rglob("*.py"))


def _violations(
    subdir: str, forbidden_prefixes: tuple[str, ...],
) -> list[tuple[Path, set[str]]]:
    out: list[tuple[Path, set[str]]] = []
    for f in _python_files(subdir):
        imports = _module_imports(f)
        bad = {m for m in imports if any(m.startswith(p) for p in forbidden_prefixes)}
        if bad:
            out.append((f, bad))
    return out


def test_cli_does_not_import_langchain_provider_packages() -> None:
    """aura/cli/** must not directly import langchain_openai/anthropic/ollama."""
    violations = _violations(
        "cli", ("langchain_openai", "langchain_anthropic", "langchain_ollama"),
    )
    assert violations == [], f"CLI-layer langchain leak: {violations}"


def test_core_does_not_import_ui_frameworks() -> None:
    """aura/core/** must not import prompt_toolkit or rich."""
    violations = _violations("core", ("prompt_toolkit", "rich"))
    assert violations == [], f"Core-layer UI-framework leak: {violations}"


# NOTE: Post-refactor (AuraTool → LangChain StructuredTool), both aura/tools/**
# and aura/cli/** legitimately import from langchain_core (BaseTool, StructuredTool).
# The old "tools-layer must not import langchain" and "cli must not import
# langchain_core" invariants no longer apply — tools *are* LangChain BaseTool
# instances by design. Those two tests were removed as architecturally obsolete.
