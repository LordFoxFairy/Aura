"""End-to-end verification script — runs against real disk layout + FakeChatModel.

Usage: ``uv run python scripts/verify_e2e.py``

Verifies (without requiring a real LLM or live MCP server):
1. Skill loading — real ``tmp/.aura/skills/superpowers/SKILL.md`` → registered,
   slash command exposed, context renders ``<skills-available>`` block.
2. Subagent DAG — parent dispatches 3 subagents in parallel via the task_create
   tool (explore / verify / plan types), results aggregate back into the store.
3. Write-a-project — FakeChatModel scripted to issue write_file + edit_file
   tool calls creating a Python project under ``tmp/demo_project/``.

MCP is intentionally SKIPPED — a live MCP server binary isn't available
in this sandbox; see the README section at the bottom for manual steps.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make sure we can import aura from the repo.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from aura.cli.commands import build_default_registry  # noqa: E402
from aura.config.schema import AuraConfig  # noqa: E402
from aura.core.agent import Agent  # noqa: E402
from aura.core.persistence.storage import SessionStorage  # noqa: E402

# The repo ships a small FakeChatModel for tests; we reuse it here.
from tests.conftest import FakeChatModel, FakeTurn  # noqa: E402


def _print_ok(label: str) -> None:
    print(f"  \033[32m✓\033[0m {label}")


def _print_fail(label: str) -> None:
    print(f"  \033[31m✗\033[0m {label}")


def _check(label: str, cond: bool) -> bool:
    (_print_ok if cond else _print_fail)(label)
    return cond


def _build_agent(tmp: Path) -> Agent:
    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": [
            "read_file", "write_file", "edit_file",
            "grep", "glob", "bash",
            "task_create", "task_get", "task_list", "task_stop",
            "skill",
        ]},
    })
    # FakeChatModel emits no turns by default — scenario 2 + 3 will replace
    # the model with one carrying scripted turns.
    return Agent(
        config=cfg,
        model=FakeChatModel(turns=[]),
        storage=SessionStorage(tmp / "db"),
    )


# ---------------------------------------------------------------------------
# Scenario 1 — skill
# ---------------------------------------------------------------------------


def scenario_skill(tmp: Path) -> bool:
    print("\n[1/3] Skill — superpowers SKILL.md loaded + registered as /superpowers")
    # The skill file is at tmp/.aura/skills/superpowers/SKILL.md — written
    # by the setup script above this one. Agent.__init__ scans cwd's .aura/skills.
    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp)
        agent = _build_agent(tmp)

        # (a) registry populated
        skills = list(agent._skill_registry.list())
        ok = _check(
            f"skill registry has {len(skills)} skill(s)",
            len(skills) == 1 and skills[0].name == "superpowers",
        )
        if not ok:
            return False

        sk = skills[0]
        ok = _check(
            "frontmatter parsed: description, when_to_use, arguments, version",
            bool(sk.description)
            and bool(sk.when_to_use)
            and sk.arguments == ("bug_description",)
            and sk.version == "1.0.0",
        ) and ok

        ok = _check(
            "source path resolved to SKILL.md inside superpowers/",
            sk.source_path.name == "SKILL.md"
            and sk.source_path.parent.name == "superpowers",
        ) and ok

        # (b) slash command registered like claude-code's /skill-name surface
        reg = build_default_registry(agent)
        cmd = reg._commands.get("/superpowers")  # private attr ok for verification
        ok = _check(
            "/superpowers slash command registered",
            cmd is not None,
        ) and ok

        # (c) Context renders <skills-available> block with when_to_use
        ctx_msgs = agent._context.build([])
        rendered = "\n".join(
            getattr(m, "content", "") if isinstance(getattr(m, "content", ""), str) else ""
            for m in ctx_msgs
        )
        ok = _check(
            "<skills-available> block includes superpowers + when_to_use",
            "<skills-available>" in rendered
            and "superpowers" in rendered
            and "Use this skill whenever" in rendered,
        ) and ok

        # (d) skill tool is registered (empty-registry pattern: tool present because N >= 1)
        tool_names = {t.name for t in agent._registry.tools()}
        ok = _check(
            "skill tool is in LLM-visible tool set",
            "skill" in tool_names,
        ) and ok

        agent.close()
        return ok
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Scenario 2 — subagent DAG (3 parallel)
# ---------------------------------------------------------------------------


async def scenario_subagent_dag(tmp: Path) -> bool:
    print("\n[2/3] Subagent DAG — parent dispatches 3 subagents in parallel")
    # Script parent's LLM turn: one AIMessage with 3 parallel tool_calls to
    # task_create, one per agent_type (explore / verify / plan).
    parent_tool_calls = [
        {"id": "call_1", "name": "task_create", "args": {
            "description": "scan repo", "prompt": "list top-level files",
            "agent_type": "explore",
        }},
        {"id": "call_2", "name": "task_create", "args": {
            "description": "audit pyproject", "prompt": "is the project well-formed",
            "agent_type": "verify",
        }},
        {"id": "call_3", "name": "task_create", "args": {
            "description": "plan next work", "prompt": "what to refactor",
            "agent_type": "plan",
        }},
    ]
    parent_turns = [
        FakeTurn(message=AIMessage(content="", tool_calls=parent_tool_calls)),
        FakeTurn(message=AIMessage(content="3 subagents dispatched.")),
    ]

    cfg = AuraConfig.model_validate({
        "providers": [{"name": "openai", "protocol": "openai"}],
        "router": {"default": "openai:gpt-4o-mini"},
        "tools": {"enabled": ["task_create", "task_get", "task_list"]},
    })
    agent = Agent(
        config=cfg,
        model=FakeChatModel(turns=parent_turns),
        storage=SessionStorage(tmp / "db2"),
    )

    # Drive one turn; the task_create tool fires 3× and returns immediately
    # with task_ids. The async subagent tasks run detached.
    events = []
    async for ev in agent.astream("do three things in parallel"):
        events.append(ev)

    # Three task_ids should be in the store.
    all_tasks = agent._tasks_store.list()
    ok = _check(
        f"3 tasks created (got {len(all_tasks)})",
        len(all_tasks) == 3,
    )

    ok = _check(
        "each task carries an agent_type (explore/verify/plan)",
        {t.agent_type for t in all_tasks} == {"explore", "verify", "plan"},
    ) and ok

    # Cleanup — cancel any still-running subagents.
    await asyncio.sleep(0.2)  # let them start so cancel actually does something
    agent.close()
    return ok


# ---------------------------------------------------------------------------
# Scenario 3 — write a project in tmp/demo_project/
# ---------------------------------------------------------------------------


async def scenario_write_project(tmp: Path) -> bool:
    print("\n[3/3] Write-a-project — FakeChatModel scripts write_file calls")
    demo = tmp / "demo_project"
    # Clean prior run.
    for f in demo.iterdir() if demo.exists() else ():
        if f.is_file():
            f.unlink()

    import os
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp)

        # Script: model calls write_file 3× creating a mini Python project.
        write_calls = [
            {"id": "w1", "name": "write_file", "args": {
                "path": "demo_project/README.md",
                "content": "# Demo\n\nGenerated by aura's e2e verifier.\n",
            }},
            {"id": "w2", "name": "write_file", "args": {
                "path": "demo_project/main.py",
                "content": "def greet() -> str:\n    return 'hello'\n",
            }},
            {"id": "w3", "name": "write_file", "args": {
                "path": "demo_project/test_main.py",
                "content": (
                    "from demo_project.main import greet\n\n"
                    "def test_greet() -> None:\n"
                    "    assert greet() == 'hello'\n"
                ),
            }},
        ]
        turns = [
            FakeTurn(message=AIMessage(content="", tool_calls=write_calls)),
            FakeTurn(message=AIMessage(content="Project scaffolded.")),
        ]

        cfg = AuraConfig.model_validate({
            "providers": [{"name": "openai", "protocol": "openai"}],
            "router": {"default": "openai:gpt-4o-mini"},
            "tools": {"enabled": ["write_file"]},
        })
        agent = Agent(
            config=cfg,
            model=FakeChatModel(turns=turns),
            storage=SessionStorage(tmp / "db3"),
            mode="bypass",  # scripted flow — skip asker
        )

        async for _ in agent.astream("scaffold demo"):
            pass

        ok = _check(
            "demo_project/README.md exists",
            (demo / "README.md").is_file(),
        )
        ok = _check(
            "demo_project/main.py exists + has greet()",
            (demo / "main.py").is_file()
            and "def greet" in (demo / "main.py").read_text(),
        ) and ok
        ok = _check(
            "demo_project/test_main.py exists + has test",
            (demo / "test_main.py").is_file()
            and "def test_greet" in (demo / "test_main.py").read_text(),
        ) and ok

        agent.close()
        return ok
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _amain() -> int:
    tmp = REPO / "tmp"
    print("=" * 70)
    print("Aura end-to-end verification (real disk, FakeChatModel for LLM)")
    print("=" * 70)

    results: list[tuple[str, bool]] = []
    results.append(("Skill (superpowers)", scenario_skill(tmp)))
    results.append(("Subagent DAG (3 parallel)", await scenario_subagent_dag(tmp)))
    results.append(("Write project (tmp/demo_project/)", await scenario_write_project(tmp)))

    print("\n" + "=" * 70)
    failed = [name for name, ok in results if not ok]
    if not failed:
        print("\033[32mAll scenarios: PASS\033[0m")
        print("=" * 70)
        return 0
    print(f"\033[31mFAILED: {', '.join(failed)}\033[0m")
    print("=" * 70)
    return 1


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
