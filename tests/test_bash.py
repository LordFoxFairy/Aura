"""Tests for aura.tools.bash — bash singleton."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aura.tools.base import AuraTool
from aura.tools.bash import BashParams, bash

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_bash_success_echo() -> None:
    result = await bash.acall(BashParams(command="echo hello"))
    assert result.ok is True
    assert result.output["stdout"] == "hello\n"
    assert result.output["exit_code"] == 0
    assert result.output["stderr"] == ""


async def test_bash_nonzero_exit_returns_ok_true() -> None:
    result = await bash.acall(BashParams(command="exit 42"))
    assert result.ok is True
    assert result.output["exit_code"] == 42


async def test_bash_nonzero_exit_captures_stderr() -> None:
    result = await bash.acall(BashParams(command="echo err >&2; exit 1"))
    assert result.ok is True
    assert "err" in result.output["stderr"]
    assert result.output["exit_code"] == 1


async def test_bash_pipe_works() -> None:
    result = await bash.acall(BashParams(command="echo hello | tr a-z A-Z"))
    assert result.ok is True
    assert "HELLO" in result.output["stdout"]


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


async def test_bash_timeout() -> None:
    result = await bash.acall(BashParams(command="sleep 5", timeout=1))
    assert result.ok is False
    assert result.error is not None
    assert "timeout" in result.error.lower()


# ---------------------------------------------------------------------------
# Capability flags / protocol
# ---------------------------------------------------------------------------


def test_bash_capability_flags() -> None:
    assert bash.is_read_only is False
    assert bash.is_destructive is True
    assert bash.is_concurrency_safe is False


def test_bash_no_check_permissions_method() -> None:
    assert not hasattr(bash, "check_permissions")


def test_bash_satisfies_protocol() -> None:
    assert isinstance(bash, AuraTool) is True


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
