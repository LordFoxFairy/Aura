"""Tests for aura.tools.bash."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.schemas.tool import ToolError
from aura.tools.bash import BashParams, bash


async def test_bash_success_echo() -> None:
    out = await bash.ainvoke({"command": "echo hello"})
    assert out["stdout"] == "hello\n"
    assert out["exit_code"] == 0
    assert out["stderr"] == ""


async def test_bash_nonzero_exit_returns_ok_true() -> None:
    out = await bash.ainvoke({"command": "exit 42"})
    assert out["exit_code"] == 42


async def test_bash_nonzero_exit_captures_stderr() -> None:
    out = await bash.ainvoke({"command": "echo err >&2; exit 1"})
    assert "err" in out["stderr"]
    assert out["exit_code"] == 1


async def test_bash_pipe_works() -> None:
    out = await bash.ainvoke({"command": "echo hello | tr a-z A-Z"})
    assert "HELLO" in out["stdout"]


async def test_bash_timeout() -> None:
    with pytest.raises(ToolError, match="timeout"):
        await bash.ainvoke({"command": "sleep 5", "timeout": 1})


def test_bash_capability_flags() -> None:
    meta = bash.metadata or {}
    assert meta.get("is_read_only") is False
    assert meta.get("is_destructive") is True
    assert meta.get("is_concurrency_safe") is False


def test_bash_no_check_permissions_method() -> None:
    assert not hasattr(bash, "check_permissions")


def test_bash_timeout_validation_rejects_too_large() -> None:
    with pytest.raises(ValidationError):
        BashParams(command="x", timeout=601)


def test_bash_timeout_validation_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        BashParams(command="x", timeout=0)


def test_bash_default_timeout() -> None:
    assert BashParams(command="x").timeout == 30


def test_bash_metadata_includes_matcher_and_preview() -> None:
    meta = bash.metadata or {}
    matcher = meta.get("rule_matcher")
    assert callable(matcher)
    # Matcher is exact-match on command.
    assert matcher({"command": "npm test"}, "npm test") is True
    assert matcher({"command": "rm -rf /"}, "npm test") is False

    preview = meta.get("args_preview")
    assert callable(preview)
    assert preview({"command": "ls"}) == "command: ls"


async def test_bash_stdout_capped_at_30k() -> None:
    # Produce 50_000 bytes of 'x' on stdout ending with a recognizable tail.
    out = await bash.ainvoke(
        {"command": "printf 'x%.0s' $(seq 1 49990); printf 'TAILMARKER'"}
    )
    stdout_bytes = out["stdout"].encode("utf-8")
    # The marker itself adds a bounded number of bytes; 200 is generous.
    assert len(stdout_bytes) <= 30_000 + 200
    assert out["stdout"].startswith("… (")
    assert out["truncated"] is True
    # Tail preserved: the final sentinel (well under 1000 bytes from end) must remain.
    assert "TAILMARKER" in out["stdout"]


async def test_bash_stderr_capped_independently() -> None:
    out = await bash.ainvoke(
        {
            "command": (
                "printf 'y%.0s' $(seq 1 49990) 1>&2; printf 'ERRTAIL' 1>&2"
            )
        }
    )
    stderr_bytes = out["stderr"].encode("utf-8")
    assert len(stderr_bytes) <= 30_000 + 200
    assert out["stderr"].startswith("… (")
    assert out["truncated"] is True
    assert out["stdout"] == ""
    assert "ERRTAIL" in out["stderr"]


async def test_bash_small_output_not_truncated() -> None:
    out = await bash.ainvoke({"command": "echo hello"})
    assert out["truncated"] is False
    assert "… (" not in out["stdout"]


async def test_bash_exactly_at_limit_not_truncated() -> None:
    # Exactly 30_000 bytes of 'x' (no trailing newline).
    out = await bash.ainvoke({"command": "printf 'x%.0s' $(seq 1 30000)"})
    assert len(out["stdout"].encode("utf-8")) == 30_000
    assert out["truncated"] is False
    assert not out["stdout"].startswith("… (")


async def test_bash_tail_preserved_in_truncation() -> None:
    out = await bash.ainvoke(
        {"command": "printf 'x%.0s' $(seq 1 50000); echo SENTINEL"}
    )
    assert out["truncated"] is True
    assert "SENTINEL" in out["stdout"]
