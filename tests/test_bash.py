"""Tests for aura.tools.bash."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.tools.base import ToolError
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
