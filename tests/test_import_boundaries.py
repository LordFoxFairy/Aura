"""Static enforcement of Aura's layer import rules."""

from __future__ import annotations

import ast
import importlib
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


def _package_for_file(py_file: Path) -> str:
    try:
        module_path = py_file.relative_to(_AURA_ROOT).with_suffix("")
        module_parts = module_path.parts
    except ValueError:
        module_path = py_file.with_suffix("")
        aura_idx = module_path.parts.index("aura")
        module_parts = module_path.parts[aura_idx:]

    if py_file.name == "__init__.py":
        return ".".join(module_parts[:-1])
    return ".".join(module_parts)


def _collect_imported_modules(py_file: Path, package: str) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is not None:
                    imported_modules.add(node.module)
                continue

            base_parts = package.split(".")
            anchor_parts = base_parts if py_file.name == "__init__.py" else base_parts[:-1]
            levels_up = node.level - 1
            if len(anchor_parts) < levels_up:
                continue

            prefix_parts = anchor_parts[: len(anchor_parts) - levels_up]
            if node.module is not None:
                target = ".".join((*prefix_parts, node.module))
                if target:
                    imported_modules.add(target)
                continue

            for alias in node.names:
                if alias.name == "*":
                    target = ".".join(prefix_parts)
                    if target:
                        imported_modules.add(target)
                    continue

                target = ".".join((*prefix_parts, alias.name))
                if target:
                    imported_modules.add(target)

                parent = ".".join(prefix_parts)
                if len(prefix_parts) > 1 and parent:
                    imported_modules.add(parent)

    return imported_modules


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


def test_new_package_roots_exist_and_import_cleanly() -> None:
    packages = (
        "aura.domain",
        "aura.runtime",
        "aura.capabilities",
        "aura.adapters",
        "aura.infrastructure",
        "aura.resources",
    )
    for pkg in packages:
        module = importlib.import_module(pkg)
        assert module is not None, f"Failed importing package root: {pkg}"


def test_resolve_imports_includes_relative_modules(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "aura" / "runtime" / "engine"
    pkg_dir.mkdir(parents=True)
    py_file = pkg_dir / "module.py"
    py_file.write_text("from .. import loop\nfrom ...adapters import http\n", encoding="utf-8")

    imports = _collect_imported_modules(py_file, "aura.runtime.engine.module")

    assert "aura.runtime" in imports
    assert "aura.adapters" in imports


def test_resolve_imports_includes_relative_modules_from_package_init(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "aura" / "runtime"
    pkg_dir.mkdir(parents=True)
    py_file = pkg_dir / "__init__.py"
    py_file.write_text("from . import loop\nfrom ..adapters import http\n", encoding="utf-8")

    imports = _collect_imported_modules(py_file, "aura.runtime")

    assert "aura.runtime" in imports
    assert "aura.adapters" in imports


def test_package_for_file_keeps_nested_package_init_path(tmp_path: Path) -> None:
    nested_init = tmp_path / "aura" / "runtime" / "engine" / "__init__.py"
    nested_init.parent.mkdir(parents=True)
    nested_init.write_text("from .. import loop\n", encoding="utf-8")

    root_init = tmp_path / "aura" / "runtime" / "__init__.py"
    root_init.parent.mkdir(parents=True, exist_ok=True)
    root_init.write_text("from . import loop\n", encoding="utf-8")

    assert _package_for_file(nested_init) == "aura.runtime.engine"
    assert _package_for_file(root_init) == "aura.runtime"


def test_resolve_imports_tracks_pure_relative_imported_names(tmp_path: Path) -> None:
    module_dir = tmp_path / "aura" / "runtime" / "engine"
    module_dir.mkdir(parents=True)

    module_file = module_dir / "module.py"
    module_file.write_text("from ... import adapters\n", encoding="utf-8")

    init_file = module_dir / "__init__.py"
    init_file.write_text("from ... import adapters\n", encoding="utf-8")

    module_imports = _collect_imported_modules(module_file, "aura.runtime.engine.module")
    init_imports = _collect_imported_modules(init_file, "aura.runtime.engine")

    assert "aura.adapters" in module_imports
    assert "aura" not in module_imports
    assert "aura.adapters" in init_imports
    assert "aura" not in init_imports


def test_layer_boundaries_for_new_roots() -> None:
    """Enforce package-level dependency direction for rearchitecture roots."""
    violations: list[tuple[str, Path, set[str]]] = []

    policy: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "domain",
            (
                "aura.runtime",
                "aura.capabilities",
                "aura.adapters",
                "aura.infrastructure",
                "aura.resources",
            ),
        ),
        ("runtime", ("aura.adapters",)),
        ("capabilities", ("aura.adapters",)),
        ("infrastructure", ("aura.runtime", "aura.capabilities", "aura.adapters")),
        (
            "resources",
            (
                "aura.domain",
                "aura.runtime",
                "aura.capabilities",
                "aura.adapters",
                "aura.infrastructure",
            ),
        ),
    )

    for subdir, forbidden in policy:
        for py_file in _python_files(subdir):
            package = _package_for_file(py_file)
            imported_modules = _collect_imported_modules(py_file, package)

            bad = {
                module
                for module in imported_modules
                if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
            }
            if bad:
                violations.append((subdir, py_file, bad))

    assert violations == [], f"New-layer dependency boundary violations: {violations}"


# NOTE: Post-refactor (AuraTool → LangChain StructuredTool), both aura/tools/**
# and aura/cli/** legitimately import from langchain_core (BaseTool, StructuredTool).
# The old "tools-layer must not import langchain" and "cli must not import
# langchain_core" invariants no longer apply — tools *are* LangChain BaseTool
# instances by design. Those two tests were removed as architecturally obsolete.
