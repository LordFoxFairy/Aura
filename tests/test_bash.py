"""Tests for aura.tools.bash — BashTool."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.tools.base import AuraTool
from aura.tools.bash import BashParams, BashTool


@pytest.fixture()
def tool() -> BashTool:
    return BashTool()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_bash_success_echo(tool: BashTool) -> None:
    result = await tool.acall(BashParams(command="echo hello"))
    assert result.ok is True
    assert result.output["stdout"] == "hello\n"
    assert result.output["exit_code"] == 0
    assert result.output["stderr"] == ""


async def test_bash_nonzero_exit_returns_ok_true(tool: BashTool) -> None:
    result = await tool.acall(BashParams(command="exit 42"))
    assert result.ok is True
    assert result.output["exit_code"] == 42


async def test_bash_nonzero_exit_captures_stderr(tool: BashTool) -> None:
    result = await tool.acall(BashParams(command="echo err >&2; exit 1"))
    assert result.ok is True
    assert "err" in result.output["stderr"]
    assert result.output["exit_code"] == 1


async def test_bash_pipe_works(tool: BashTool) -> None:
    result = await tool.acall(BashParams(command="echo hello | tr a-z A-Z"))
    assert result.ok is True
    assert "HELLO" in result.output["stdout"]


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


async def test_bash_timeout(tool: BashTool) -> None:
    result = await tool.acall(BashParams(command="sleep 5", timeout=1))
    assert result.ok is False
    assert result.error is not None
    assert "timeout" in result.error.lower()


# ---------------------------------------------------------------------------
# Capability flags / protocol
# ---------------------------------------------------------------------------


def test_bash_capability_flags(tool: BashTool) -> None:
    assert tool.is_read_only is False
    assert tool.is_destructive is True
    assert tool.is_concurrency_safe is False


def test_bash_no_check_permissions_method(tool: BashTool) -> None:
    assert not hasattr(tool, "check_permissions")


def test_bash_satisfies_protocol(tool: BashTool) -> None:
    assert isinstance(tool, AuraTool) is True


# ---------------------------------------------------------------------------
# Param validation
# ---------------------------------------------------------------------------


def test_bash_timeout_validation_rejects_too_large() -> None:
    with pytest.raises(ValidationError):
        BashParams(command="x", timeout=601)


def test_bash_timeout_validation_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        BashParams(command="x", timeout=0)


def test_bash_default_timeout() -> None:
    assert BashParams(command="x").timeout == 30
