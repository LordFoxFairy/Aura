"""Subprocess entrypoint smoke tests.

The pane backend launches teammates via
``python -m aura.cli.teammate_entrypoint``; this suite verifies the
module is importable, the argparse front matches the documented shape,
and ``--help`` runs cleanly. Real subprocess round-trip is covered by
the pane backend tests (which require an actual tmux session).
"""

from __future__ import annotations

import argparse
import importlib

import pytest


def test_teammate_entrypoint_imports_clean() -> None:
    """Module imports without side effects (config load, agent build, etc.)."""
    mod = importlib.import_module("aura.cli.teammate_entrypoint")
    assert hasattr(mod, "main")
    assert hasattr(mod, "_make_parser")


def test_teammate_entrypoint_parser_required_flags() -> None:
    """``--team-id``, ``--member``, ``--storage-root`` are all required."""
    from aura.cli.teammate_entrypoint import _make_parser

    parser = _make_parser()
    # Missing required flags -> argparse exits 2.
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args([
        "--team-id", "t1",
        "--member", "alice",
        "--storage-root", "/tmp/aura",
    ])
    assert args.team_id == "t1"
    assert args.member == "alice"
    assert args.storage_root == "/tmp/aura"
    # Defaults for optional flags.
    assert args.agent_type == "general-purpose"
    assert args.model is None
    assert args.system_prompt is None
    assert args.seed_prompt is None


def test_teammate_entrypoint_parser_optional_flags() -> None:
    """Optional flags carry through to the parsed Namespace."""
    from aura.cli.teammate_entrypoint import _make_parser

    parser = _make_parser()
    args = parser.parse_args([
        "--team-id", "t1",
        "--member", "alice",
        "--storage-root", "/tmp/aura",
        "--agent-type", "researcher",
        "--model", "openai:gpt-4o",
        "--system-prompt", "You are alice",
        "--seed-prompt", "Find the bug",
    ])
    assert args.agent_type == "researcher"
    assert args.model == "openai:gpt-4o"
    assert args.system_prompt == "You are alice"
    assert args.seed_prompt == "Find the bug"


def test_teammate_entrypoint_help_runs_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` prints to stdout and exits zero (argparse convention)."""
    from aura.cli.teammate_entrypoint import _make_parser

    parser = _make_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "team-id" in captured.out
    assert "member" in captured.out
    assert "storage-root" in captured.out


def test_teammate_entrypoint_main_function_signature() -> None:
    """``main`` accepts an optional argv list and returns an int."""
    # Verify signature shape — the caller passes ``argv: list[str] | None``.
    import inspect

    from aura.cli.teammate_entrypoint import main

    sig = inspect.signature(main)
    assert "argv" in sig.parameters
    # Default is None so ``main()`` parses sys.argv.
    assert sig.parameters["argv"].default is None
    # Return annotation is int (string form due to ``from __future__ import
    # annotations`` deferring evaluation).
    assert sig.return_annotation in (int, "int")


def test_argparse_parser_matches_pane_backend_command_shape() -> None:
    """Pane backend's argv must be parseable by the entrypoint parser.

    Smoke test that ``PaneBackend._build_subprocess_command`` produces
    a list of args this parser accepts. Catches drift if either side
    renames a flag without the other.
    """
    from pathlib import Path

    from aura.cli.teammate_entrypoint import _make_parser
    from aura.core.persistence.storage import SessionStorage
    from aura.core.teams.backends.pane import PaneBackend
    from aura.core.teams.types import TeammateMember

    storage = SessionStorage(Path(":memory:"))
    member = TeammateMember(
        name="alice",
        agent_type="general-purpose",
        model_name="openai:gpt-4o",
        backend_type="pane",
    )
    argv = PaneBackend._build_subprocess_command(
        team_id="t1",
        member=member,
        storage=storage,
        seed_prompt="hi there",
    )
    # Drop ``python -m aura.cli.teammate_entrypoint`` (first 3 elements);
    # the entrypoint module is invoked via ``-m`` so argparse sees the
    # tail as its argv.
    parser = _make_parser()
    parsed = parser.parse_args(argv[3:])
    assert parsed.team_id == "t1"
    assert parsed.member == "alice"
    assert parsed.agent_type == "general-purpose"
    assert parsed.model == "openai:gpt-4o"
    assert parsed.seed_prompt == "hi there"


def test_main_module_invocation_form() -> None:
    """``python -m aura.cli.teammate_entrypoint`` resolves to a real module."""
    import importlib.util

    spec = importlib.util.find_spec("aura.cli.teammate_entrypoint")
    assert spec is not None
    assert spec.origin is not None
    assert isinstance(_make_parser_top := importlib.import_module(
        "aura.cli.teammate_entrypoint",
    )._make_parser, type(lambda: None))
    parser = _make_parser_top()
    assert isinstance(parser, argparse.ArgumentParser)
