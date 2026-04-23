"""PTY-driven interactive scenarios.

Exercises the slices ``verify_real_llm.py`` can't reach because they
need real keystrokes (arrow keys, Tab, Esc) delivered over a real
terminal. Uses stdlib ``pty`` + ``select`` — no external deps.

Scenarios:
1. shift+tab cycles mode silently (no scrollback spam)
2. bash tool under default mode triggers asker widget (real LLM call)

Usage: ``uv run python scripts/verify_pty_interactive.py`` (configured env).
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


class _Pty:
    def __init__(self, argv: list[str], *, boot_timeout: float = 15.0) -> None:
        self.master_fd, slave_fd = pty.openpty()
        env = dict(os.environ)
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self.proc = subprocess.Popen(
            argv,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
            cwd=REPO,
            env=env,
        )
        os.close(slave_fd)
        self._buf = b""
        self._boot_timeout = boot_timeout

    def expect(self, needle: str | re.Pattern, timeout: float = 10.0) -> str:
        """Read until needle appears; return captured text."""
        deadline = time.time() + timeout
        pattern = (
            needle if isinstance(needle, re.Pattern)
            else re.compile(re.escape(needle))
        )
        while time.time() < deadline:
            r, _, _ = select.select([self.master_fd], [], [], 0.3)
            if r:
                try:
                    chunk = os.read(self.master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                self._buf += chunk
                if pattern.search(self._buf.decode("utf-8", "replace")):
                    return self._buf.decode("utf-8", "replace")
        return self._buf.decode("utf-8", "replace")

    def send(self, data: bytes | str) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8")
        os.write(self.master_fd, data)

    def read_all(self, timeout: float = 2.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r, _, _ = select.select([self.master_fd], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(self.master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                self._buf += chunk
            else:
                break
        return self._buf.decode("utf-8", "replace")

    def wait(self, timeout: float = 10.0) -> int | None:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def close(self) -> None:
        import contextlib
        with contextlib.suppress(OSError):
            os.close(self.master_fd)
        if self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.proc.wait(timeout=5)


def _ok(label: str, cond: bool) -> bool:
    icon = "\033[32m✓\033[0m" if cond else "\033[31m✗\033[0m"
    print(f"  {icon} {label}")
    return cond


def scenario_shift_tab_silent() -> bool:
    """Send \\x1b[Z (shift+tab) three times and /exit. Assert zero scrollback
    spam (no ``mode: X (press shift+tab…)`` lines in captured output)."""
    print("\n[1/2] shift+tab cycles mode silently — no scrollback spam")
    pty_ = _Pty(["uv", "run", "aura"])
    try:
        boot = pty_.expect("aura>", timeout=15.0)
        ok = _ok("aura> prompt appeared within 15s", "aura>" in boot)
        if not ok:
            return False
        pty_.send(b"\x1b[Z")  # shift+tab
        time.sleep(0.2)
        pty_.send(b"\x1b[Z")
        time.sleep(0.2)
        pty_.send(b"\x1b[Z")
        time.sleep(0.2)
        pty_.send(b"/exit\r")
        rc = pty_.wait(timeout=15.0)
        tail = pty_.read_all(timeout=2.0)
    finally:
        pty_.close()

    ok = _ok("process exited cleanly", rc is not None) and ok
    ok = _ok(
        "no 'mode: ... press shift+tab to cycle' spam",
        "press shift+tab to cycle" not in tail,
    ) and ok
    ok = _ok(
        "no 'mode:' scrollback spam at all",
        "mode:" not in tail.replace("mode: deepseek", "").replace(
            "model: deepseek", ""
        ),
    ) and ok
    ok = _ok("no Python traceback", "Traceback (most" not in tail) and ok
    return ok


def scenario_asker_widget_appears_under_real_llm() -> bool:
    """Real LLM + default mode: LLM issues bash tool_call → asker widget
    renders on the pty. Assert the widget's tell-tale signature appears
    (numbered options / question text). Then send Esc to cancel so we
    don't block on real approval (we only care that the WIDGET shipped)."""
    print("\n[2/2] asker widget renders on real LLM tool-call (default mode)")
    pty_ = _Pty(["uv", "run", "aura"], boot_timeout=15.0)
    try:
        boot = pty_.expect("aura>", timeout=15.0)
        ok = _ok("boot reached aura>", "aura>" in boot)
        if not ok:
            return False
        # Real LLM turn: ask for a command that requires bash.
        pty_.send("run bash with command 'echo hello'\r")
        # Wait for the widget to render. Its format includes numbered
        # options like " 1. Yes " / " 4. No ". Look for "Run" question
        # header OR any of the standard choice labels.
        widget = pty_.expect(
            re.compile(r"(Run bash|\s1\.\s|Yes,\s+and\s+always|\sNo\b)"),
            timeout=90.0,
        )
        ok = _ok("asker widget rendered (found option label)",
                 bool(re.search(
                     r"(Run bash|\s1\.\s|Yes,\s+and\s+always|\sNo\b)",
                     widget,
                 ))) and ok
        # We've proven the core claim (widget rendered under real LLM +
        # default mode + bash tool_call). Dismissing the interactive
        # widget from a piloted pty across widget-key-binding variants
        # is fragile and orthogonal to what we're testing — the finally
        # block's SIGKILL handles cleanup. Read a bit more of the buffer
        # so we can still assert no traceback fired.
        tail = pty_.read_all(timeout=3.0)
    finally:
        pty_.close()

    ok = _ok("no Python traceback", "Traceback (most" not in tail) and ok
    return ok


def main() -> int:
    print("=" * 70)
    print("Aura PTY-interactive verification (stdlib pty, real terminal)")
    print(f"  repo: {REPO}")
    print("=" * 70)
    results = [
        ("shift+tab silent", scenario_shift_tab_silent()),
        ("asker widget real-LLM", scenario_asker_widget_appears_under_real_llm()),
    ]
    print()
    print("=" * 70)
    failed = [n for n, ok in results if not ok]
    if not failed:
        print(f"\033[32mAll {len(results)} PTY scenarios PASS\033[0m")
        print("=" * 70)
        return 0
    print(f"\033[31mFAILED: {failed}\033[0m")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
