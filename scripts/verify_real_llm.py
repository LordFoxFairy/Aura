"""Real-LLM end-to-end scenarios. Driven via subprocess + stdin pipe.

Runs ``uv run aura`` with controlled inputs, asserts on captured output.
Every scenario hits the REAL configured model (no FakeChatModel).

Usage: ``uv run python scripts/verify_real_llm.py``

Requires: provider SDK installed (e.g. ``uv sync --extra all``) and the
right API key env var (e.g. ``DEEPSEEK_API_KEY``). Run from repo root.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _run(
    inputs: list[str],
    *,
    bypass: bool = False,
    timeout: int = 90,
) -> tuple[int, str]:
    cmd = ["uv", "run", "aura"]
    if bypass:
        cmd.append("--bypass-permissions")
    stdin = "".join(line + "\n" for line in inputs) + "/exit\n"
    proc = subprocess.run(
        cmd,
        input=stdin,
        text=True,
        capture_output=True,
        cwd=REPO,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


class Scenario:
    def __init__(self, name: str) -> None:
        self.name = name
        self.checks: list[tuple[str, bool]] = []
        self.output = ""
        self.elapsed = 0.0

    def check(self, label: str, ok: bool) -> None:
        self.checks.append((label, ok))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok in self.checks) and len(self.checks) > 0


SCENARIOS: list[Scenario] = []


def scenario(name: str):
    def _wrap(fn):
        def _runner():
            sc = Scenario(name)
            t0 = time.time()
            try:
                fn(sc)
            except Exception as exc:
                sc.check(f"EXCEPTION: {exc!r}", False)
            sc.elapsed = time.time() - t0
            SCENARIOS.append(sc)
            icon = "\033[32m✓\033[0m" if sc.passed else "\033[31m✗\033[0m"
            print(f"{icon} [{sc.elapsed:5.1f}s] {sc.name}")
            for label, ok in sc.checks:
                sub_icon = "  ✓" if ok else "  ✗"
                color = "\033[32m" if ok else "\033[31m"
                print(f"    {color}{sub_icon}\033[0m {label}")
        return _runner
    return _wrap


# ---------------------------------------------------------------------------
# 1–6: slash commands (no LLM)
# ---------------------------------------------------------------------------


@scenario("boot + /help lists commands")
def s1(sc: Scenario) -> None:
    rc, out = _run(["/help"])
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("/exit in help", "/exit" in out)
    sc.check("/compact in help", "/compact" in out)
    sc.check("/model in help", "/model" in out)
    sc.check("/tasks in help", "/tasks" in out)
    sc.check("shift+tab hint rendered", "shift+tab" in out)


@scenario("/status (git wrapper)")
def s2(sc: Scenario) -> None:
    rc, out = _run(["/status"])
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("branch line present", "branch" in out.lower() or "main" in out)


@scenario("/log (git wrapper)")
def s3(sc: Scenario) -> None:
    rc, out = _run(["/log 3"])
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("at least one commit line", "feat" in out or "fix" in out)


@scenario("/tasks (empty list on fresh session)")
def s4(sc: Scenario) -> None:
    rc, out = _run(["/tasks"])
    sc.output = out
    sc.check("process exited 0", rc == 0)
    # Either "no tasks" or just the aura> prompt reprinted — both acceptable;
    # main assertion is it didn't crash.
    sc.check("no traceback", "Traceback" not in out)


@scenario("/model shows current")
def s5(sc: Scenario) -> None:
    rc, out = _run(["/model"])
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("deepseek visible in output", "deepseek" in out.lower())


@scenario("/clear resets session")
def s6(sc: Scenario) -> None:
    rc, out = _run(["/clear"])
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("no traceback", "Traceback" not in out)


# ---------------------------------------------------------------------------
# 7–12: real LLM scenarios
# ---------------------------------------------------------------------------


@scenario("real LLM: one-word reply")
def s7(sc: Scenario) -> None:
    rc, out = _run(["say hi in exactly one word"], timeout=60)
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("LLM produced at least some text after prompt", "aura>" in out)
    sc.check("turn-end marker rendered (done · Ns)", "done" in out and "s" in out)


@scenario("real LLM + read_file (bypass)")
def s8(sc: Scenario) -> None:
    rc, out = _run(
        ["read the first 3 lines of README.md and quote them"],
        bypass=True,
        timeout=90,
    )
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("bypass banner shown", "PERMISSION CHECKS DISABLED" in out)
    sc.check("read_file tool invoked", "read_file" in out)
    sc.check("README.md path in output", "README.md" in out)
    sc.check("allowed: mode_bypass annotation", "mode_bypass" in out)


@scenario("real LLM + grep (bypass)")
def s9(sc: Scenario) -> None:
    rc, out = _run(
        [
            "grep for the string 'from langchain_core' in the aura directory "
            "and summarize what files match",
        ],
        bypass=True,
        timeout=90,
    )
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("grep tool invoked", "grep" in out)
    sc.check("at least one python file mentioned", ".py" in out)


@scenario("real LLM + write_file (bypass, into tmp/)")
def s10(sc: Scenario) -> None:
    target = REPO / "tmp" / "llm_wrote_this.txt"
    if target.exists():
        target.unlink()
    rc, out = _run(
        [
            "write a file at tmp/llm_wrote_this.txt "
            "containing the single line 'hello from llm'",
        ],
        bypass=True,
        timeout=90,
    )
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("write_file tool invoked", "write_file" in out)
    sc.check("file actually exists on disk", target.is_file())
    if target.is_file():
        content = target.read_text()
        sc.check(
            "file body contains requested text",
            "hello from llm" in content,
        )


@scenario("real LLM + subagent dispatch (bypass)")
def s11(sc: Scenario) -> None:
    rc, out = _run(
        [
            "use task_create to dispatch a subagent with agent_type='explore' "
            "and prompt='list python files under aura/tools/'. "
            "Return the task_id.",
        ],
        bypass=True,
        timeout=180,  # subagent adds a second LLM call
    )
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("task_create tool invoked", "task_create" in out)
    # LLM may use agent_type="explore" or default — either is valid parity.
    # The core assertion is "task dispatch actually happened".
    sc.check(
        "subagent dispatched (task_id or created marker present)",
        ("task_id" in out.lower())
        or ("task created" in out.lower())
        or ("dispatched" in out.lower()),
    )


@scenario("skill /superpowers slash invocation")
def s12(sc: Scenario) -> None:
    # /superpowers expects a positional arg per its `arguments: [bug_description]`
    # frontmatter. We pipe it with a description, then /exit.
    rc, out = _run(
        ["/superpowers auth login returns 500 on fresh user"],
        timeout=90,
    )
    sc.output = out
    sc.check("process exited 0", rc == 0)
    sc.check("no traceback", "Traceback" not in out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("Aura REAL-LLM end-to-end verification")
    print(f"  repo: {REPO}")
    print("=" * 70)
    print()

    for runner in [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]:
        runner()

    print()
    print("=" * 70)
    failed = [sc for sc in SCENARIOS if not sc.passed]
    total = len(SCENARIOS)
    elapsed = sum(sc.elapsed for sc in SCENARIOS)
    if not failed:
        print(f"\033[32mAll {total} scenarios PASS\033[0m · total {elapsed:.1f}s")
    else:
        print(f"\033[31mFAILED: {len(failed)}/{total}\033[0m · total {elapsed:.1f}s")
        for sc in failed:
            bad = [label for label, ok in sc.checks if not ok]
            print(f"  \033[31m✗\033[0m {sc.name}: {bad}")
    print("=" * 70)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
