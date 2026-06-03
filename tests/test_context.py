"""Tests for aura.application.memory.context: 4-layer assembly and progressive state."""

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

from aura.application.memory.context import (
    Context,
    NestedFragment,
)
from aura.application.memory.context import _render_todos_body as render_todos_body
from aura.application.memory.context_types import ReadRecord as ContextReadRecord
from aura.application.memory.rules_types import Rule, RulesBundle
from aura.domain.skill import Skill
from aura.domain.state_values import ReadCarryover, ReadRecord
from aura.domain.task import TaskNotification
from aura.domain.todos import TodoItem


def _rule(source: Path, base_dir: Path, globs: tuple[str, ...], body: str) -> Rule:
    return Rule(
        source_path=source.resolve() if source.exists() else source,
        base_dir=base_dir,
        globs=globs,
        content=body,
    )


def _skill(name: str, desc: str = "desc", body: str = "body") -> Skill:
    return Skill(
        name=name,
        description=desc,
        body=body,
        source_path=Path(f"/tmp/{name}.md"),
        layer="user",
    )


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
    # F-03-003: project-memory now emits as a SystemMessage wrapped in a
    # <system-reminder> envelope with an OVERRIDE preamble.
    assert isinstance(out[1], SystemMessage)
    body = str(out[1].content)
    assert body.startswith("<system-reminder>\n")
    assert body.endswith("\n</system-reminder>")
    assert "OVERRIDE" in body
    assert (
        "<project-memory>\nPRIMARY\n\nRULE-1-BODY\n\nRULE-2-BODY\n</project-memory>"
    ) in body


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
    assert "<project-memory>\nA\n\nB\n</project-memory>" in str(out[1].content)


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
    # Same path touched twice emits a single nested fragment.
    assert len(out) == 2


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
    # Paths outside cwd must not contribute a nested fragment.
    assert len(out) == 1
    assert isinstance(out[0], SystemMessage)


def test_07_nested_walk_outer_before_inner_cwd_excluded(tmp_path: Path) -> None:
    cwd = tmp_path / "p"
    a = cwd / "a"
    b = a / "b"
    b.mkdir(parents=True)
    # The cwd-level AURA.md is eager and must not be re-included as nested.
    (cwd / "AURA.md").write_text("CWD-MEMO")
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
    # Expected: [SystemMessage, nested(a/AURA.md), nested(a/b/AURA.md)].
    assert len(out) == 3
    a_path = (a / "AURA.md").resolve()
    b_path = (b / "AURA.md").resolve()
    assert str(a_path) in out[1].content
    assert "A-MEMO" in out[1].content
    assert str(b_path) in out[2].content
    assert "B-MEMO" in out[2].content
    # cwd-level AURA.md must not appear as a nested fragment.
    assert "CWD-MEMO" not in out[1].content
    assert "CWD-MEMO" not in out[2].content


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
    # Expected: [SystemMessage, rule] — no subdir AURA.md so no nested fragment.
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
    # A rule matched by multiple paths still emits a single HumanMessage.
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

    # Bundle insertion order is reversed; match() sorts alphabetically, so
    # the output order should be a/b/c.
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
    # Expected: System + Layer2 + 2 nested + 3 rule + 4 history = 11.
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
    # ctx_a: System + nested == 2 messages. ctx_b: System only == 1 message.
    assert len(out_a) == 2
    assert len(out_b) == 1


def test_nested_fragment_is_frozen_dataclass(tmp_path: Path) -> None:
    frag = NestedFragment(source=tmp_path / "x.md", content="X")
    with pytest.raises(FrozenInstanceError):
        frag.__setattr__("content", "Y")


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
        TodoItem(content="a", status="pending", active_form="Doing a"),
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

    # Exactly one todos message.
    todos_idxs = [
        i for i, m in enumerate(out) if str(m.content).startswith("<todos>\n")
    ]
    assert len(todos_idxs) == 1
    todos_idx = todos_idxs[0]
    todos_msg = out[todos_idx]
    assert isinstance(todos_msg, HumanMessage)
    assert str(todos_msg.content).startswith("<todos>\n")
    assert str(todos_msg.content).endswith("\n</todos>")

    # The todos message sits after every <rule> message and before history.
    rule_idxs = [i for i, m in enumerate(out) if "<rule " in str(m.content)]
    assert rule_idxs
    assert todos_idx > max(rule_idxs)
    assert out[todos_idx + 1] is history[0]
    assert out[todos_idx + 2] is history[1]


def test_ac11_todos_body_contains_item_fields(tmp_path: Path) -> None:
    todos = [
        TodoItem(content="parse config", status="pending", active_form="Parsing config"),
        TodoItem(
            content="write tests",
            status="in_progress",
            active_form="Writing tests",
        ),
        TodoItem(
            content="scaffold module", status="completed", active_form="Scaffolding"
        ),
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
    # Each item's content and status appears; activeForm appears for non-completed items.
    assert "parse config" in body
    assert "pending" in body
    assert "Parsing config" in body
    assert "write tests" in body
    assert "in_progress" in body
    assert "Writing tests" in body
    assert "scaffold module" in body
    assert "completed" in body


def test_ac12_no_todos_provider_means_no_todos_message(tmp_path: Path) -> None:
    # todos_provider kwarg omitted entirely.
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    out = ctx.build([])
    assert all("<todos>" not in str(m.content) for m in out)

    # Explicit None should behave the same.
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

    todos = [TodoItem(content="t1", status="pending", active_form="Doing t1")]

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

    # Expected: 1 system + 1 project-memory + 2 nested + 3 rules + 1 todos + 4 history = 12.
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


def test_context_full_section_order_includes_skills_todos_notifications_before_history(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "p"
    src = cwd / "src"
    src.mkdir(parents=True)
    (src / "AURA.md").write_text("SRC-MEMO")
    touched = src / "x.py"
    touched.write_text("")

    rule = _rule(
        tmp_path / "rules" / "py.md",
        cwd.resolve(),
        ("**/*.py",),
        "PY-RULE",
    )
    skill = _skill("helper", "helps", "HELPER-BODY")
    todos = [
        TodoItem(
            content="Write draft",
            status="in_progress",
            active_form="Writing draft",
        )
    ]
    notification = TaskNotification(
        task_id="task-123456",
        status="completed",
        description="child task",
        summary="child done",
    )
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="PRIMARY",
        rules=RulesBundle(conditional=[rule]),
        skills=[skill],
        todos_provider=lambda: todos,
        notifications_drainer=lambda: [notification],
    )
    ctx.on_tool_touched_path(touched)
    ctx.record_skill_invocation(skill)
    history: list[BaseMessage] = [HumanMessage("user"), AIMessage("assistant")]

    out = ctx.build(history)
    contents = [str(m.content) for m in out]

    assert contents[0] == "SYS"
    assert "<project-memory>" in contents[1]
    assert contents[2].startswith("<nested-memory ")
    assert "SRC-MEMO" in contents[2]
    assert contents[3].startswith("<rule ")
    assert "PY-RULE" in contents[3]
    assert contents[4].startswith("<skills-available>")
    assert "- helper: helps" in contents[4]
    assert contents[5].startswith('<skill-invoked name="helper">')
    assert "HELPER-BODY" in contents[5]
    assert contents[6].startswith("<todos>")
    assert "Write draft" in contents[6]
    assert contents[7].startswith("<task-notification>")
    assert "task-123" in contents[7]
    assert "child done" in contents[7]
    assert out[8] is history[0]
    assert out[9] is history[1]


def test_context_task_notifications_cap_and_drain_once(tmp_path: Path) -> None:
    queue = [
        TaskNotification(
            task_id=f"task{i:04d}-id",
            status="completed",
            description=f"task {i}",
            summary=f"summary {i}",
        )
        for i in range(8)
    ]

    def drain() -> list[TaskNotification]:
        drained = list(queue)
        queue.clear()
        return drained

    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        notifications_drainer=drain,
    )

    first = ctx.build([])
    notification = next(
        str(m.content)
        for m in first
        if str(m.content).startswith("<task-notification>")
    )
    assert "task0000" not in notification
    assert "task0003" in notification
    assert "task0007" in notification
    assert "(3 more earlier)" in notification

    second = ctx.build([])
    assert not any(
        str(m.content).startswith("<task-notification>") for m in second
    )


def test_context_read_status_stale_after_size_change_and_delete(
    tmp_path: Path,
) -> None:
    target = tmp_path / "note.txt"
    target.write_text("one")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )

    ctx.record_read(target)
    assert ctx.read_status(target) == "fresh"
    target.write_text("one two")
    assert ctx.read_status(target) == "stale"

    ctx.record_read(target)
    assert ctx.read_status(target) == "fresh"
    target.unlink()
    assert ctx.read_status(target) == "stale"


def test_ac17_auto_clear_roundtrip_provider_returning_empty_list(
    tmp_path: Path,
) -> None:
    # Simulates post-auto-clear state: the provider snapshots
    # ``state.slots.todos`` after the tool has reset it to [].
    store: dict[str, list[TodoItem]] = {"todos": []}
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
        todos_provider=lambda: store["todos"],
    )
    out = ctx.build([])
    assert all("<todos>" not in str(m.content) for m in out)


# The renderer lives in context.py (not tools/) to keep the core → tools
# dependency direction clean.


def test_render_todos_body_non_completed_includes_active_form() -> None:
    body = render_todos_body(
        [TodoItem(content="a", status="pending", active_form="Doing a")]
    )
    assert "a" in body
    assert "pending" in body
    assert "Doing a" in body


def test_render_todos_body_completed_omits_active_form() -> None:
    body = render_todos_body(
        [TodoItem(content="done-task", status="completed", active_form="Did it")]
    )
    assert "done-task" in body
    assert "completed" in body


def test_render_todos_body_multi_line_no_trailing_newline() -> None:
    body = render_todos_body(
        [
            TodoItem(content="a", status="pending", active_form="Doing a"),
            TodoItem(content="b", status="completed", active_form="Did b"),
        ]
    )
    assert body.count("\n") == 1
    assert not body.endswith("\n")


def test_fresh_clears_progressive_state(tmp_path: Path) -> None:
    """fresh() returns a NEW Context with progressive fields cleared."""
    cwd = tmp_path / "p"
    src = cwd / "src"
    src.mkdir(parents=True)
    (src / "AURA.md").write_text("SRC-MEMO")
    py_file = src / "x.py"
    py_file.write_text("")

    rule = _rule(
        tmp_path / "rules" / "py.md",
        cwd.resolve(),
        ("**/*.py",),
        "PY-RULE",
    )
    skill = _skill("helper", "helps", "HELPER-BODY")
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(conditional=[rule]),
        skills=[skill],
    )
    # Populate every progressive field.
    ctx.on_tool_touched_path(py_file)
    ctx.record_skill_invocation(skill)
    assert ctx._loaded_nested_paths
    assert ctx._nested_fragments
    assert ctx._matched_rule_paths
    assert ctx._matched_rules
    assert ctx._invoked_skill_paths
    assert ctx._invoked_skills

    new_ctx = ctx.fresh()

    # New instance, not the same object.
    assert new_ctx is not ctx
    # Every progressive field reset.
    assert new_ctx._loaded_nested_paths == set()
    assert new_ctx._nested_fragments == []
    assert new_ctx._matched_rule_paths == set()
    assert new_ctx._matched_rules == []
    assert new_ctx._invoked_skill_paths == set()
    assert new_ctx._invoked_skills == []
    # The original instance is untouched.
    assert ctx._loaded_nested_paths
    assert ctx._invoked_skills


def test_fresh_preserves_config(tmp_path: Path) -> None:
    """fresh() carries constructor-injected config onto the new instance."""
    skill = _skill("doc", "documents", "DOC-BODY")
    todos_provider = lambda: [  # noqa: E731  # lambda in test is fine
        TodoItem(content="t", status="pending", active_form="Doing t"),
    ]
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS-PROMPT",
        primary_memory="PRIMARY-MEM",
        rules=RulesBundle(),
        skills=[skill],
        todos_provider=todos_provider,
    )
    new_ctx = ctx.fresh()

    # Config preserved by value-equality on the message stream.
    out = new_ctx.build([])
    contents = [str(m.content) for m in out]
    assert contents[0] == "SYS-PROMPT"
    assert "PRIMARY-MEM" in contents[1]
    # <skills-available> still rendered — skills list survived.
    assert any(c.startswith("<skills-available>") for c in contents)
    assert any("- doc: documents" in c for c in contents)
    # todos_provider survived (renders the pending todo).
    assert any(c.startswith("<todos>") for c in contents)


def test_fresh_with_carryover_seeds_read_records(tmp_path: Path) -> None:
    """fresh(carryover=...) seeds _read_records from a ReadCarryover."""
    target = tmp_path / "note.txt"
    target.write_text("hello")
    stat = target.stat()

    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    record = ReadRecord(
        path=target,
        mtime_at_read=stat.st_mtime,
        size_at_read=stat.st_size,
        read_at_turn=3,
    )
    carry = ReadCarryover(
        records={target: record},
        source_session_id="parent-1",
        generated_at_turn=3,
    )
    new_ctx = ctx.fresh(carryover=carry)

    assert target in new_ctx._read_records
    seeded = new_ctx._read_records[target]
    assert isinstance(seeded, ContextReadRecord)
    assert seeded.mtime == stat.st_mtime
    assert seeded.size == stat.st_size
    assert seeded.partial is False
    # The seeded record satisfies read_status (file unchanged on disk).
    assert new_ctx.read_status(target) == "fresh"


def test_fresh_with_clear_reads_drops_read_records(tmp_path: Path) -> None:
    """fresh(clear_reads=True) wipes the _read_records map."""
    target = tmp_path / "note.txt"
    target.write_text("hello")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.record_read(target)
    assert ctx.read_status(target) == "fresh"

    new_ctx = ctx.fresh(clear_reads=True)
    assert new_ctx._read_records == {}
    assert new_ctx.read_status(target) == "never_read"
    # Original instance untouched.
    assert ctx.read_status(target) == "fresh"


def test_fresh_default_preserves_read_records(tmp_path: Path) -> None:
    """fresh() with no args preserves _read_records (file survives compact)."""
    target = tmp_path / "note.txt"
    target.write_text("hello")
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.record_read(target)
    new_ctx = ctx.fresh()
    assert new_ctx.read_status(target) == "fresh"
    # Independent maps — recording on the new instance doesn't pollute old.
    other = tmp_path / "other.txt"
    other.write_text("x")
    new_ctx.record_read(other)
    assert ctx.read_status(other) == "never_read"


def test_fresh_rejects_carryover_and_clear_reads_together(tmp_path: Path) -> None:
    """Passing both carryover and clear_reads is a programmer error."""
    ctx = Context(
        cwd=tmp_path,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    carry = ReadCarryover(
        records={},
        source_session_id=None,
        generated_at_turn=0,
    )
    with pytest.raises(ValueError, match="not both"):
        ctx.fresh(carryover=carry, clear_reads=True)


def test_path_in_scope_helper_under_cwd(tmp_path: Path) -> None:
    """``_path_in_scope`` returns True for paths at or below cwd."""
    cwd = tmp_path / "p"
    inside = cwd / "src" / "x.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    assert ctx._path_in_scope(inside) is True
    assert ctx._path_in_scope(cwd) is True


def test_path_in_scope_helper_outside_cwd(tmp_path: Path) -> None:
    """``_path_in_scope`` returns False for paths outside cwd."""
    cwd = tmp_path / "p"
    cwd.mkdir()
    outside = tmp_path / "outside" / "y.py"
    outside.parent.mkdir()
    outside.write_text("")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    assert ctx._path_in_scope(outside) is False


def test_out_of_cwd_path_skips_rule_match(tmp_path: Path) -> None:
    """Phase 3 §5: out-of-cwd paths must not trigger conditional rule match."""
    cwd = tmp_path / "p"
    cwd.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    py_outside = outside / "x.py"
    py_outside.write_text("")

    # User-layer style rule: base_dir = home, glob = "**/*.py" (matches any .py).
    # Pre-fix: this would match an out-of-cwd path. Post-fix: it must not.
    rule = _rule(
        tmp_path / "rules" / "py.md",
        tmp_path,
        ("**/*.py",),
        "USER-PY-BODY",
    )
    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(unconditional=[], conditional=[rule]),
    )
    ctx.on_tool_touched_path(py_outside)
    out = ctx.build([])
    # No rule match triggered — only the SystemMessage should remain.
    assert all("<rule " not in str(m.content) for m in out)
    assert ctx._matched_rules == []


def test_out_of_cwd_path_skips_nested_memory(tmp_path: Path) -> None:
    """Phase 3 §5: out-of-cwd paths must not trigger nested-memory load.

    Re-affirms the existing `test_06` invariant alongside the rule-match
    one, so the unified behaviour is locked in test-by-test.
    """
    cwd = tmp_path / "p"
    cwd.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "AURA.md").write_text("OUTSIDE-MEMO")
    py_outside = outside / "x.py"
    py_outside.write_text("")

    ctx = Context(
        cwd=cwd,
        system_prompt="SYS",
        primary_memory="",
        rules=RulesBundle(),
    )
    ctx.on_tool_touched_path(py_outside)
    assert ctx._nested_fragments == []
    out = ctx.build([])
    assert all("<nested-memory " not in str(m.content) for m in out)
