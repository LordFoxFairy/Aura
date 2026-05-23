"""Subagent type registry — the 4 flavors exposed via ``task_create(agent_type=...)``.

Covers the pure registry surface (``get_agent_def`` / ``all_agent_defs``)
and the intrinsic shape of each built-in type. Integration with the factory
+ tool lives in ``test_task_tools.py`` and ``test_tasks_factory.py``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from aura.infrastructure.agents import (
    AgentDef,
    all_agent_defs,
    get_agent_def,
)


def test_get_agent_def_returns_def_for_each_name() -> None:
    for name in ("general-purpose", "explore", "verify", "plan"):
        td = get_agent_def(name)
        assert isinstance(td, AgentDef)
        assert td.name == name


def test_get_agent_def_unknown_raises_with_valid_names_listed() -> None:
    with pytest.raises(ValueError) as ei:
        get_agent_def("bogus")
    msg = str(ei.value)
    # Every valid name must appear in the error message so the LLM (which
    # sees this via ToolError) can self-correct without another round-trip.
    for name in ("general-purpose", "explore", "verify", "plan"):
        assert name in msg


def test_all_agent_defs_returns_all_four_in_declaration_order() -> None:
    got = [td.name for td in all_agent_defs()]
    # Built-ins must lead in declaration order; user-defined files may
    # follow but the four built-ins always come first.
    assert got[:4] == ["general-purpose", "explore", "verify", "plan"]


def test_general_purpose_has_empty_tools_inherit_all_sentinel() -> None:
    td = get_agent_def("general-purpose")
    # Empty frozenset is the documented "inherit all from parent" sentinel —
    # distinct from any concrete allowlist.
    assert td.tools == frozenset()
    assert td.system_prompt_suffix == ""


def test_explore_tools_exclude_writes_and_recursion() -> None:
    td = get_agent_def("explore")
    # Read-only contract: no writes, no shell, no nested dispatch.
    forbidden = {
        "write_file",
        "edit_file",
        "bash",
        "bash_background",
        "task_create",
        "task_output",
        "task_stop",
    }
    assert not (td.tools & forbidden)
    # Positive allowlist must carry the declared five.
    assert {"read_file", "grep", "glob", "web_fetch", "web_search"} <= td.tools
    assert "Explore" in td.system_prompt_suffix


def test_verify_prompt_contains_verdict_marker() -> None:
    td = get_agent_def("verify")
    # The strict output contract is enforced via the prompt — if the marker
    # string drifts, the parent's downstream parser will silently fail.
    assert "VERDICT:" in td.system_prompt_suffix
    assert "PASS" in td.system_prompt_suffix
    assert "FAIL" in td.system_prompt_suffix
    # Read-only like explore.
    assert "write_file" not in td.tools
    assert "bash" not in td.tools


def test_plan_tools_include_both_plan_mode_controls() -> None:
    td = get_agent_def("plan")
    # Plan subagent must be able to both enter AND exit plan mode — without
    # exit_plan_mode it can never hand a plan back to the parent.
    assert "enter_plan_mode" in td.tools
    assert "exit_plan_mode" in td.tools
    # Still read-only otherwise — no shell, no writes, no recursion.
    assert "write_file" not in td.tools
    assert "edit_file" not in td.tools
    assert "bash" not in td.tools
    assert "task_create" not in td.tools


def test_agent_def_is_immutable() -> None:
    # Frozen dataclass + frozenset inner: attempting to mutate either the
    # def or its tools must raise. Keeps the registry safe from accidental
    # in-place edits by callers.
    td = get_agent_def("explore")
    with pytest.raises(FrozenInstanceError):
        # Reassignment of a frozen field must raise at runtime. Route
        # through setattr (not ``td.name = ...``) so mypy doesn't reject
        # the narrow-Literal mismatch — the point of this test is the
        # runtime frozen contract, not the static type check.
        setattr(td, "name", "hacked")  # noqa: B010
    with pytest.raises(AttributeError):
        td.tools.add("bash")  # type: ignore[attr-defined]  # test sets attribute mypy can't see
