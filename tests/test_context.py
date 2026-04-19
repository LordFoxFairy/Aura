"""Tests for aura.core.context (Task 5): 4-layer assembly + progressive state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from aura.core.context import Context, NestedFragment
from aura.core.rules import Rule, RulesBundle


def _rule(source: Path, base_dir: Path, globs: tuple[str, ...], body: str) -> Rule:
    return Rule(
        source_path=source.resolve() if source.exists() else source,
        base_dir=base_dir,
        globs=globs,
        content=body,
    )


# ---------------------------------------------------------------------------
# Construction + eager (B8) — 3 tests
# ---------------------------------------------------------------------------


def test_01_eager_layer2_with_primary_and_two_unconditional_rules(
    tmp_path: Path,
) -> None:
    r1 = _rule(tmp_path / "u1.md", tmp_path, (), "RULE-1-BODY")
    r2 = _rule(tmp_path / "u2.md", tmp_path, (), "RULE-2-BODY")
    bundle = RulesBundle(unconditional=[r1, r2], conditional=[])

    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=bundle,
    )
    out = ctx.build([])
    assert len(out) == 2
    assert isinstance(out[0], SystemMessage)
    assert out[0].content == "SYS"
    assert isinstance(out[1], HumanMessage)
    assert out[1].content == (
        "<project-memory>\n"
        "PRIMARY\n\n"
        "RULE-1-BODY\n\n"
        "RULE-2-BODY\n"
        "</project-memory>"
    )


def test_02_empty_primary_and_empty_bundle_emits_system_only(
    tmp_path: Path,
) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    out = ctx.build([])
    assert len(out) == 1
    assert isinstance(out[0], SystemMessage)


def test_03_only_unconditional_rules_layer2_contains_rules_only(
    tmp_path: Path,
) -> None:
    r1 = _rule(tmp_path / "u1.md", tmp_path, (), "A")
    r2 = _rule(tmp_path / "u2.md", tmp_path, (), "B")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[r1, r2], conditional=[]),
    )
    out = ctx.build([])
    assert len(out) == 2
    assert out[1].content == "<project-memory>\nA\n\nB\n</project-memory>"


# ---------------------------------------------------------------------------
# Progressive subdir walk-down (B3) — 4 tests
# ---------------------------------------------------------------------------


def test_04_subdir_aura_md_loaded_on_tool_touched_path(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    src = cwd / "src"
    src.mkdir(parents=True)
    (src / "AURA.md").write_text("SRC-MEMO")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.on_tool_touched_path(src / "foo.py")
    out = ctx.build([])
    # [SystemMessage, nested-memory]
    assert len(out) == 2
    nested = out[1]
    assert isinstance(nested, HumanMessage)
    expected_path = (src / "AURA.md").resolve()
    assert nested.content == (
        f'<nested-memory path="{expected_path}">\n'
        "SRC-MEMO\n"
        "</nested-memory>"
    )


def test_05_same_path_touched_twice_dedup(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    src = cwd / "src"
    src.mkdir(parents=True)
    (src / "AURA.md").write_text("SRC")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.on_tool_touched_path(src / "foo.py")
    ctx.on_tool_touched_path(src / "foo.py")
    out = ctx.build([])
    assert len(out) == 2  # one nested fragment only


def test_06_path_outside_cwd_no_fragment(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    cwd.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "AURA.md").write_text("IGNORED")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.on_tool_touched_path(outside / "x.py")
    out = ctx.build([])
    # SystemMessage only — no nested fragment added.
    assert len(out) == 1
    assert isinstance(out[0], SystemMessage)


def test_07_nested_walk_outer_before_inner_cwd_excluded(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    a = cwd / "a"
    b = a / "b"
    b.mkdir(parents=True)
    (cwd / "AURA.md").write_text("CWD-MEMO")  # eager; should NOT be re-included
    (a / "AURA.md").write_text("A-MEMO")
    (b / "AURA.md").write_text("B-MEMO")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.on_tool_touched_path(b / "c.py")
    out = ctx.build([])
    # [SystemMessage, nested(a/AURA.md), nested(a/b/AURA.md)]  — no cwd-level nested
    assert len(out) == 3
    a_path = (a / "AURA.md").resolve()
    b_path = (b / "AURA.md").resolve()
    assert str(a_path) in out[1].content
    assert "A-MEMO" in out[1].content
    assert str(b_path) in out[2].content
    assert "B-MEMO" in out[2].content
    # Ensure cwd AURA.md did NOT get injected as a nested fragment.
    assert "CWD-MEMO" not in out[1].content
    assert "CWD-MEMO" not in out[2].content


# ---------------------------------------------------------------------------
# Progressive rules (B6) — 3 tests
# ---------------------------------------------------------------------------


def test_08_conditional_rule_triggered_by_path(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    (cwd / "src").mkdir(parents=True)
    py_file = cwd / "src" / "x.py"
    py_file.write_text("")

    rule = _rule(
        tmp_path / "rules" / "py.md",
        cwd.resolve(),
        ("**/*.py",),
        "PY-RULE-BODY",
    )
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[rule]),
    )
    ctx.on_tool_touched_path(py_file)
    out = ctx.build([])
    # [SystemMessage, rule] (no nested since no subdir AURA.md)
    assert len(out) == 2
    assert isinstance(out[1], HumanMessage)
    assert f'<rule src="{rule.source_path}">' in out[1].content
    assert "PY-RULE-BODY" in out[1].content


def test_09_same_rule_matched_by_multiple_paths_dedup(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    (cwd / "src").mkdir(parents=True)
    p1 = cwd / "src" / "a.py"
    p2 = cwd / "src" / "b.py"
    p1.write_text("")
    p2.write_text("")

    rule = _rule(
        tmp_path / "rules" / "py.md",
        cwd.resolve(),
        ("**/*.py",),
        "BODY",
    )
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[rule]),
    )
    ctx.on_tool_touched_path(p1)
    ctx.on_tool_touched_path(p2)
    out = ctx.build([])
    # Only one rule HumanMessage.
    rule_msgs = [m for m in out if "<rule " in str(m.content)]
    assert len(rule_msgs) == 1


def test_10_matched_rules_sorted_by_source_path(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    (cwd / "src").mkdir(parents=True)
    p_py = cwd / "src" / "x.py"
    p_py.write_text("")

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    ra = _rule(rules_dir / "a.md", cwd.resolve(), ("**/*.py",), "A-BODY")
    rb = _rule(rules_dir / "b.md", cwd.resolve(), ("**/*.py",), "B-BODY")
    rc = _rule(rules_dir / "c.md", cwd.resolve(), ("**/*.py",), "C-BODY")

    # Insertion order in bundle reversed — match() sorts, so output is a/b/c.
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[rc, rb, ra]),
    )
    ctx.on_tool_touched_path(p_py)
    out = ctx.build([])
    rule_msgs = [m for m in out if "<rule " in str(m.content)]
    assert len(rule_msgs) == 3
    assert "A-BODY" in rule_msgs[0].content
    assert "B-BODY" in rule_msgs[1].content
    assert "C-BODY" in rule_msgs[2].content


# ---------------------------------------------------------------------------
# Mixed order (B8) — 1 test
# ---------------------------------------------------------------------------


def test_11_full_build_order_system_layer2_nested_rules_history(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "p"
    a = cwd / "a"
    b = a / "b"
    b.mkdir(parents=True)
    (a / "AURA.md").write_text("A-MEMO")
    (b / "AURA.md").write_text("B-MEMO")

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    ra = _rule(rules_dir / "a.md", cwd.resolve(), ("**/*.py",), "RA")
    rb = _rule(rules_dir / "b.md", cwd.resolve(), ("**/*.py",), "RB")
    rc = _rule(rules_dir / "c.md", cwd.resolve(), ("**/*.py",), "RC")

    touched = b / "x.py"
    touched.write_text("")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[ra, rb, rc]),
    )
    ctx.on_tool_touched_path(touched)
    history: list[BaseMessage] = [
        HumanMessage("u1"),
        AIMessage("a1"),
        HumanMessage("u2"),
        AIMessage("a2"),
    ]
    out = ctx.build(history)
    # Expected: System + Layer2 + 2×nested + 3×rule + 4×history = 11
    assert len(out) == 11
    assert isinstance(out[0], SystemMessage)
    assert "<project-memory>" in str(out[1].content)
    assert "A-MEMO" in str(out[2].content)
    assert "B-MEMO" in str(out[3].content)
    assert "RA" in str(out[4].content)
    assert "RB" in str(out[5].content)
    assert "RC" in str(out[6].content)
    assert out[7] is history[0]
    assert out[8] is history[1]
    assert out[9] is history[2]
    assert out[10] is history[3]


# ---------------------------------------------------------------------------
# Fresh instance invariant — 1 test
# ---------------------------------------------------------------------------


def test_12_two_instances_independent_progressive_state(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    (cwd / "src").mkdir(parents=True)
    (cwd / "src" / "AURA.md").write_text("SRC")

    bundle = RulesBundle()
    ctx_a = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=bundle,
    )
    ctx_b = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=bundle,
    )
    ctx_a.on_tool_touched_path(cwd / "src" / "foo.py")
    out_a = ctx_a.build([])
    out_b = ctx_b.build([])
    # a: System + nested — 2 messages.  b: System only — 1 message.
    assert len(out_a) == 2
    assert len(out_b) == 1


def test_nested_fragment_is_frozen_dataclass(tmp_path: Path) -> None:
    frag = NestedFragment(source=tmp_path / "x.md", content="X")
    with pytest.raises(FrozenInstanceError):
        frag.content = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# <todos> emission (Phase C-2 Task 2) — AC 9, 10, 11, 12, 16, 17
# ---------------------------------------------------------------------------


def test_ac09_empty_todos_list_emits_no_todos_message(tmp_path: Path) -> None:
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        todos_provider=lambda: [],
    )
    out = ctx.build([])
    assert all("<todos>" not in str(m.content) for m in out)


def test_ac10_non_empty_todos_emits_single_todos_humanmessage_after_rules(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "p"
    (cwd / "src").mkdir(parents=True)
    p_py = cwd / "src" / "x.py"
    p_py.write_text("")
    rule = _rule(
        tmp_path / "rules" / "py.md",
        cwd.resolve(),
        ("**/*.py",),
        "PY-RULE",
    )
    todos = [
        {"content": "a", "status": "pending", "activeForm": "Doing a"},
    ]
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[rule]),
        todos_provider=lambda: todos,
    )
    ctx.on_tool_touched_path(p_py)
    history: list[BaseMessage] = [HumanMessage("u1"), AIMessage("a1")]
    out = ctx.build(history)

    # Find the todos message — exactly one.
    todos_idxs = [
        i for i, m in enumerate(out) if str(m.content).startswith("<todos>\n")
    ]
    assert len(todos_idxs) == 1
    todos_idx = todos_idxs[0]
    todos_msg = out[todos_idx]
    assert isinstance(todos_msg, HumanMessage)
    assert str(todos_msg.content).startswith("<todos>\n")
    assert str(todos_msg.content).endswith("\n</todos>")

    # Sits after every <rule> message and before history.
    rule_idxs = [i for i, m in enumerate(out) if "<rule " in str(m.content)]
    assert rule_idxs  # sanity
    assert todos_idx > max(rule_idxs)
    # History tail starts right after todos.
    assert out[todos_idx + 1] is history[0]
    assert out[todos_idx + 2] is history[1]


def test_ac11_todos_body_contains_item_fields(tmp_path: Path) -> None:
    todos = [
        {"content": "parse config", "status": "pending", "activeForm": "Parsing config"},
        {
            "content": "write tests",
            "status": "in_progress",
            "activeForm": "Writing tests",
        },
        {"content": "scaffold module", "status": "completed", "activeForm": "Scaffolding"},
    ]
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        todos_provider=lambda: todos,
    )
    out = ctx.build([])
    todos_msg = next(m for m in out if str(m.content).startswith("<todos>\n"))
    body = str(todos_msg.content)
    # Each item's content + status shows up; activeForm shows for non-completed.
    assert "parse config" in body
    assert "pending" in body
    assert "Parsing config" in body
    assert "write tests" in body
    assert "in_progress" in body
    assert "Writing tests" in body
    assert "scaffold module" in body
    assert "completed" in body


def test_ac12_no_todos_provider_means_no_todos_message(tmp_path: Path) -> None:
    # Omit kwarg entirely.
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    out = ctx.build([])
    assert all("<todos>" not in str(m.content) for m in out)

    # Explicit None.
    ctx_none = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        todos_provider=None,
    )
    out_none = ctx_none.build([])
    assert all("<todos>" not in str(m.content) for m in out_none)


def test_ac16_full_build_ordering_primary_nested_rules_todos_history(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "p"
    a = cwd / "a"
    b = a / "b"
    b.mkdir(parents=True)
    (a / "AURA.md").write_text("A-MEMO")
    (b / "AURA.md").write_text("B-MEMO")

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    ra = _rule(rules_dir / "a.md", cwd.resolve(), ("**/*.py",), "RA")
    rb = _rule(rules_dir / "b.md", cwd.resolve(), ("**/*.py",), "RB")
    rc = _rule(rules_dir / "c.md", cwd.resolve(), ("**/*.py",), "RC")

    touched = b / "x.py"
    touched.write_text("")

    todos = [{"content": "t1", "status": "pending", "activeForm": "Doing t1"}]

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(unconditional=[], conditional=[ra, rb, rc]),
        todos_provider=lambda: todos,
    )
    ctx.on_tool_touched_path(touched)
    history: list[BaseMessage] = [
        HumanMessage("u1"),
        AIMessage("a1"),
        HumanMessage("u2"),
        AIMessage("a2"),
    ]
    out = ctx.build(history)

    # 1 system + 1 project-memory + 2 nested + 3 rules + 1 todos + 4 history = 12.
    assert len(out) == 12
    assert isinstance(out[0], SystemMessage)
    assert "<project-memory>" in str(out[1].content)
    assert "A-MEMO" in str(out[2].content)
    assert "B-MEMO" in str(out[3].content)
    assert "RA" in str(out[4].content)
    assert "RB" in str(out[5].content)
    assert "RC" in str(out[6].content)
    assert str(out[7].content).startswith("<todos>\n")
    assert str(out[7].content).endswith("\n</todos>")
    assert "t1" in str(out[7].content)
    assert out[8] is history[0]
    assert out[9] is history[1]
    assert out[10] is history[2]
    assert out[11] is history[3]


def test_ac17_auto_clear_roundtrip_provider_returning_empty_list(
    tmp_path: Path,
) -> None:
    # Simulate the post-auto-clear state: provider snapshots state.custom["todos"]
    # which has been reset to [] by the tool.
    store: dict[str, list[dict[str, str]]] = {"todos": []}
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        todos_provider=lambda: store["todos"],
    )
    out = ctx.build([])
    assert all("<todos>" not in str(m.content) for m in out)
