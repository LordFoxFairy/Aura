"""``cli teammate`` subcommand smoke tests.

The pane backend launches teammates via ``python -m cli teammate``;
this suite verifies the argparse front matches the documented shape and
that the pane backend's argv builder lines up with the subcommand's
parser. Real subprocess round-trip is covered by the pane backend tests
(which require an actual tmux session).
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pytest

from cli.__main__ import _make_parser, run_as_teammate


def test_teammate_subcommand_required_flags() -> None:
    """``--team-id``, ``--member``, ``--storage-root`` are all required."""
    parser = _make_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["teammate"])
    args = parser.parse_args([
        "teammate",
        "--team-id", "t1",
        "--member", "alice",
        "--storage-root", "/tmp/aura",
    ])
    assert args.subcommand == "teammate"
    assert args.team_id == "t1"
    assert args.member == "alice"
    assert args.storage_root == "/tmp/aura"
    assert args.agent_type == "general-purpose"
    assert args.model is None
    assert args.system_prompt is None
    assert args.seed_prompt is None


def test_teammate_subcommand_optional_flags() -> None:
    """Optional flags carry through to the parsed Namespace."""
    parser = _make_parser()
    args = parser.parse_args([
        "teammate",
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


def test_teammate_subcommand_help_runs_clean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``teammate --help`` prints to stdout and exits zero."""
    parser = _make_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["teammate", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "team-id" in captured.out
    assert "member" in captured.out
    assert "storage-root" in captured.out


def test_run_as_teammate_signature() -> None:
    """``run_as_teammate`` takes a Namespace and returns an int."""
    sig = inspect.signature(run_as_teammate)
    assert "args" in sig.parameters
    assert sig.return_annotation in (int, "int")


def test_argparse_parser_matches_pane_backend_command_shape() -> None:
    """Pane backend's argv must be parseable by the ``teammate`` subcommand.

    Smoke test that ``PaneBackend._build_subprocess_command`` produces
    a list of args this parser accepts. Catches drift if either side
    renames a flag without the other.
    """
    from aura.domain.team import TeammateMember
    from aura.infrastructure.persistence.storage import SessionStorage
    from aura.infrastructure.teams.pane import PaneBackend

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
    # argv is ``[python, -m, cli, teammate, --team-id, ...]``; drop the
    # interpreter + ``-m cli`` prefix so the parser sees the same tail
    # the cli would see during a real spawn.
    parser = _make_parser()
    parsed = parser.parse_args(argv[3:])
    assert parsed.subcommand == "teammate"
    assert parsed.team_id == "t1"
    assert parsed.member == "alice"
    assert parsed.agent_type == "general-purpose"
    assert parsed.model == "openai:gpt-4o"
    assert parsed.seed_prompt == "hi there"


def test_make_parser_returns_argparse_parser() -> None:
    """``_make_parser`` returns a usable :class:`argparse.ArgumentParser`."""
    parser = _make_parser()
    assert isinstance(parser, argparse.ArgumentParser)
