"""Tests for cli.__main__ entry point."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_version_flag_fast_path() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aura" in result.stdout.lower()


def test_help_flag() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aura" in result.stdout.lower()


def test_plaintext_api_key_emits_warning_only_in_verbose(tmp_path: Path) -> None:
    # Non-verbose runs SHOULD stay silent — the warning fires every startup
    # otherwise and operators tune it out (which defeats the point of the
    # warning in the first place). Verbose still prints for operators
    # actively debugging or auditing.
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "providers": [{
            "name": "test",
            "protocol": "openai",
            "api_key": "sk-plaintext-secret",
        }],
        "router": {"default": "test:gpt-4o-mini"},
    }))

    from rich.console import Console

    from aura.config.loader import load_config
    from cli.__main__ import _warn_plaintext_api_keys

    cfg = load_config(user_config=config_path, project_config=tmp_path / "absent.json")

    # Default (no --verbose) — silent.
    buf_quiet = io.StringIO()
    _warn_plaintext_api_keys(
        cfg, Console(file=buf_quiet, force_terminal=False, width=200),
    )
    assert "Warning" not in buf_quiet.getvalue()
    assert "plaintext" not in buf_quiet.getvalue()

    # --verbose — prints.
    buf_verbose = io.StringIO()
    _warn_plaintext_api_keys(
        cfg, Console(file=buf_verbose, force_terminal=False, width=200),
        verbose=True,
    )
    out = buf_verbose.getvalue()
    assert "Warning" in out
    assert "'test'" in out
    assert "plaintext" in out


def test_plaintext_api_key_writes_journal_event(tmp_path: Path) -> None:
    # Troubleshooting contract: the console warning is ``--verbose`` only,
    # so the journal entry MUST fire unconditionally — that's what makes
    # the audit trail reliable. An operator grepping events.jsonl for
    # "why did this provider expose a key" must find a machine-readable
    # record regardless of whether anyone was watching the console.
    import pytest
    from rich.console import Console

    from aura.config.loader import load_config
    from aura.infrastructure.persistence import journal
    from cli.__main__ import _warn_plaintext_api_keys

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "providers": [
            {"name": "alpha", "protocol": "openai", "api_key": "sk-1"},
            {"name": "beta", "protocol": "openai", "api_key_env": "BETA_KEY"},
            {"name": "gamma", "protocol": "openai", "api_key": "sk-2"},
        ],
        "router": {"default": "alpha:gpt-4o-mini"},
    }))
    cfg = load_config(user_config=config_path, project_config=tmp_path / "absent.json")

    events: list[tuple[str, dict[str, object]]] = []

    def _capture(event: str, /, **fields: object) -> None:
        events.append((event, fields))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(journal, "write", _capture)
        _warn_plaintext_api_keys(
            cfg, Console(file=io.StringIO(), force_terminal=False, width=200),
        )

    plaintext_events = [e for e in events if e[0] == "plaintext_api_key_warning"]
    assert len(plaintext_events) == 2  # alpha + gamma, beta uses api_key_env
    names = {e[1]["provider"] for e in plaintext_events}
    assert names == {"alpha", "gamma"}


def _ns(**kw: object) -> argparse.Namespace:
    import argparse

    return argparse.Namespace(**kw)


def test_resolve_mode_defaults_to_default() -> None:
    # Post-2026-04-21: _resolve_mode takes a PermissionsConfig, not AuraConfig.
    # No flag + default PermissionsConfig (mode="default") → "default".
    from aura.config.schema import PermissionsConfig
    from cli.__main__ import _resolve_mode

    args = _ns(bypass_permissions=False)
    # deliberately off-type arg to exercise path
    assert _resolve_mode(args, PermissionsConfig()) == "default"


def test_resolve_mode_reads_permissions_config_mode() -> None:
    # PermissionsConfig comes from settings.json (via store.load), not from
    # AuraConfig. Mode set there should be honored when the flag is off.
    from aura.config.schema import PermissionsConfig
    from cli.__main__ import _resolve_mode

    perm_cfg = PermissionsConfig(mode="bypass")
    args = _ns(bypass_permissions=False)
    # deliberately off-type arg to exercise path
    assert _resolve_mode(args, perm_cfg) == "bypass"


def test_resolve_mode_cli_flag_wins_over_settings_default() -> None:
    from aura.config.schema import PermissionsConfig
    from cli.__main__ import _resolve_mode

    perm_cfg = PermissionsConfig(mode="default")
    args = _ns(bypass_permissions=True)
    # deliberately off-type arg to exercise path
    assert _resolve_mode(args, perm_cfg) == "bypass"


def test_resolve_mode_cli_flag_wins_even_over_settings_bypass() -> None:
    # Trivial but worth locking: flag True always wins regardless of the
    # settings value. (A user could explicitly set bypass in both places;
    # ordering must be predictable.)
    from aura.config.schema import PermissionsConfig
    from cli.__main__ import _resolve_mode

    perm_cfg = PermissionsConfig(mode="bypass")
    args = _ns(bypass_permissions=True)
    # deliberately off-type arg to exercise path
    assert _resolve_mode(args, perm_cfg) == "bypass"


def test_bypass_refused_message_is_stable() -> None:
    # The error message is the single piece of text the operator sees
    # when their --bypass-permissions attempt gets refused. Lock its
    # shape so docs / support runbooks can reference it.
    from cli.__main__ import _bypass_refused_message

    msg = _bypass_refused_message()
    assert "--bypass-permissions is disabled" in msg
    assert "disable_bypass=true" in msg


def test_bypass_refused_end_to_end_via_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Finding B acceptance: --bypass-permissions + disable_bypass=true
    # must exit 2 (config error) and print the refused message to
    # stderr. We exercise main() in-process (subprocess-based tests
    # misroute to a stale parent venv in editable-install setups)
    # with HOME + cwd scoped to tmp_path and the LLM factory
    # monkeypatched out so startup never touches real providers.
    from cli.__main__ import main

    # Minimal user config so load_config succeeds.
    user_aura_dir = tmp_path / ".aura"
    user_aura_dir.mkdir()
    (user_aura_dir / "config.json").write_text(json.dumps({
        "providers": [{
            "name": "p1",
            "protocol": "openai",
            "api_key_env": "FAKE_API_KEY",
        }],
        "router": {"default": "p1:fake-model"},
    }))
    # Project settings.json sets the kill switch.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / ".aura").mkdir()
    (project_dir / ".aura" / "settings.json").write_text(json.dumps({
        "permissions": {"disable_bypass": True},
    }))

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura", "--bypass-permissions"])

    rc = main()
    captured = capsys.readouterr()
    assert rc == 2, (
        f"expected exit 2, got {rc}; "
        f"stdout={captured.out!r} stderr={captured.err!r}"
    )
    assert "--bypass-permissions is disabled" in captured.err
    assert "disable_bypass=true" in captured.err


def test_disable_bypass_false_allows_bypass_flag() -> None:
    # Regression lock: when disable_bypass is false (the default), the
    # --bypass-permissions flag still yields mode="bypass" via
    # _resolve_mode. We exercise the resolver directly because the full
    # main() path requires an LLM client; the kill-switch check lives
    # AFTER _resolve_mode in main() and is covered by the subprocess
    # test above.
    from aura.config.schema import PermissionsConfig
    from cli.__main__ import _resolve_mode

    perm_cfg = PermissionsConfig(disable_bypass=False)
    args = _ns(bypass_permissions=True)
    # deliberately off-type arg to exercise path
    assert _resolve_mode(args, perm_cfg) == "bypass"
    assert perm_cfg.disable_bypass is False


def test_main_wires_allow_deny_and_ask_rules_into_permission_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aura.application.hooks import HookChain
    from cli.__main__ import main

    user_aura_dir = tmp_path / ".aura"
    user_aura_dir.mkdir()
    (user_aura_dir / "config.json").write_text(json.dumps({
        "providers": [{
            "name": "p1",
            "protocol": "openai",
            "api_key_env": "FAKE_API_KEY",
        }],
        "router": {"default": "p1:fake-model"},
        "tools": {"enabled": ["web_fetch", "write_file", "bash"]},
    }))
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / ".aura").mkdir()
    (project_dir / ".aura" / "settings.json").write_text(json.dumps({
        "permissions": {
            "allow": ["web_fetch"],
            "deny": ["bash"],
            "ask": ["write_file"],
        },
    }))

    captured_hook_kwargs: dict[str, object] = {}
    captured_build_kwargs: dict[str, object] = {}

    def fake_permission_hook(**kwargs: object) -> object:
        captured_hook_kwargs.update(kwargs)

        async def _hook(**_kw: object) -> object:
            raise AssertionError("permission hook should not run in this test")

        return _hook

    class FakeAgent:
        mode = "default"
        state = object()
        _hooks = HookChain()
        hooks = _hooks

        async def aconnect(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_build_agent(*_args: object, **kwargs: object) -> FakeAgent:
        captured_build_kwargs.update(kwargs)
        return FakeAgent()

    class FakeWatcher:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    async def fake_repl(*_args: object, **_kwargs: object) -> None:
        return None

    # Patch where main() looks them up: top-level imports bind into cli.__main__.
    import cli.__main__ as main_mod
    import cli.repl as repl_mod

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura"])
    monkeypatch.setattr(main_mod, "make_permission_hook", fake_permission_hook)
    monkeypatch.setattr(main_mod, "build_agent", fake_build_agent)
    monkeypatch.setattr(main_mod, "FileWatcher", FakeWatcher)
    monkeypatch.setattr(repl_mod, "run_repl_async", fake_repl)

    assert main() == 0

    from aura.domain.permission.session import RuleSet

    rules_obj = captured_hook_kwargs["rules"]
    deny_rules_obj = captured_hook_kwargs["deny_rules"]
    ask_rules_obj = captured_hook_kwargs["ask_rules"]
    assert isinstance(rules_obj, RuleSet)
    assert isinstance(deny_rules_obj, RuleSet)
    assert isinstance(ask_rules_obj, RuleSet)
    assert [rule.tool for rule in rules_obj.rules][:1] == ["web_fetch"]
    assert [rule.tool for rule in deny_rules_obj.rules] == ["bash"]
    assert [rule.tool for rule in ask_rules_obj.rules] == ["write_file"]
    assert captured_build_kwargs["ruleset"] is rules_obj
    assert captured_build_kwargs["deny_ruleset"] is deny_rules_obj
    assert captured_build_kwargs["ask_ruleset"] is ask_rules_obj


# --------------------------------------------------------------------------
# Boundary tests (appended) — exercise dispatch, error, and wiring branches
# that the subprocess/resolver tests above leave uncovered. Everything runs
# in-process with the heavy seams (build_agent / run_repl_async / FileWatcher
# / run_teammate_main / handle_mcp) monkeypatched, so no real session, model,
# subprocess, or network is touched.
# --------------------------------------------------------------------------

from collections.abc import Callable  # noqa: E402

from rich.console import Console  # noqa: E402

import cli.__main__ as _main_mod  # noqa: E402
import cli.repl as _repl_mod  # noqa: E402
from aura.application.hooks import HookChain  # noqa: E402
from aura.domain.errors import AuraError  # noqa: E402


def _silent_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=200)


class _FakeAgent:
    """Stand-in for AgentSession so main() never builds a real model."""

    mode = "default"
    state = object()
    hooks = HookChain()

    async def aconnect(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeWatcher:
    """No-op FileWatcher: never touches the filesystem or spawns watchers."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _write_min_layout(tmp_path: Path, *, settings: dict[str, object]) -> Path:
    """Lay down a minimal $HOME config + project settings; return project dir."""
    user_aura_dir = tmp_path / ".aura"
    user_aura_dir.mkdir()
    (user_aura_dir / "config.json").write_text(json.dumps({
        "providers": [{
            "name": "p1",
            "protocol": "openai",
            "api_key_env": "FAKE_API_KEY",
        }],
        "router": {"default": "p1:fake-model"},
    }))
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / ".aura").mkdir()
    (project_dir / ".aura" / "settings.json").write_text(
        json.dumps({"permissions": settings}),
    )
    return project_dir


def _wire_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    project_dir: Path,
    *,
    argv: list[str],
    agent_factory: Callable[[], _FakeAgent] | None = None,
    repl: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Patch every heavy seam in cli.__main__ and return captured hook kwargs.

    The returned dict is filled by a fake permission hook, exposing the
    live-mode closure and rule objects main() wired up.
    """
    captured: dict[str, object] = {}

    def fake_permission_hook(**kwargs: object) -> object:
        captured.update(kwargs)

        async def _hook(**_kw: object) -> object:
            raise AssertionError("permission hook should not run")

        return _hook

    make_agent = agent_factory if agent_factory is not None else _FakeAgent

    def fake_build_agent(*_args: object, **_kwargs: object) -> _FakeAgent:
        return make_agent()

    async def default_repl(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(_main_mod, "make_permission_hook", fake_permission_hook)
    monkeypatch.setattr(_main_mod, "build_agent", fake_build_agent)
    monkeypatch.setattr(_main_mod, "FileWatcher", _FakeWatcher)
    monkeypatch.setattr(
        _repl_mod, "run_repl_async", repl if repl is not None else default_repl,
    )
    return captured


# ---- _split_dashdash --------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["a", "b"], (["a", "b"], [])),
        (["a", "--", "b", "c"], (["a"], ["b", "c"])),
        (["--"], ([], [])),
        (["--", "x"], ([], ["x"])),
        (["a", "--", "--", "b"], (["a"], ["--", "b"])),
    ],
)
def test_split_dashdash_partition(
    argv: list[str], expected: tuple[list[str], list[str]],
) -> None:
    # `aura mcp add NAME -- cmd args` relies on this split to separate
    # Aura flags from the server's own command vector. The first `--`
    # is the boundary; everything after (even a second `--`) belongs to
    # the server command and must pass through untouched.
    from cli.__main__ import _split_dashdash

    assert _split_dashdash(argv) == expected


# ---- _fail_startup ----------------------------------------------------------


def test_fail_startup_aura_error_uses_typed_message_and_journals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An AuraError is a known/expected failure: the operator should see the
    # exception class name (not a generic "startup error"), and the journal
    # must record reason=<ClassName> so support can triage from events.jsonl.
    from aura.infrastructure.persistence import journal

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        journal, "write",
        lambda event, /, **f: events.append((event, f)),
    )
    buf = io.StringIO()
    rc = _main_mod._fail_startup(
        Console(file=buf, force_terminal=False, width=200), AuraError("boom"),
    )
    assert rc == 2
    assert "AuraError: boom" in buf.getvalue()
    assert events == [
        ("startup_failed", {"reason": "AuraError", "detail": "boom"}),
    ]


def test_fail_startup_unexpected_error_is_labelled_and_journaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-AuraError is an *unexpected* crash: the console gets a generic
    # "startup error" line and the journal tags reason="unexpected" so these
    # are distinguishable from known config/domain failures during triage.
    from aura.infrastructure.persistence import journal

    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        journal, "write",
        lambda event, /, **f: events.append((event, f)),
    )
    buf = io.StringIO()
    rc = _main_mod._fail_startup(
        Console(file=buf, force_terminal=False, width=200), ValueError("oops"),
    )
    assert rc == 2
    assert "startup error: oops" in buf.getvalue()
    assert events == [
        ("startup_failed", {"reason": "unexpected", "detail": "oops"}),
    ]


# ---- run_as_teammate --------------------------------------------------------


def _teammate_ns(**over: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "team_id": "T",
        "member": "M",
        "storage_root": "/tmp/aura-teammate",
        "agent_type": "general-purpose",
        "model": None,
        "system_prompt": None,
        "seed_prompt": None,
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_run_as_teammate_forwards_namespace_and_returns_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The teammate subprocess entrypoint must pass argv fields straight
    # through to run_teammate_main and surface its exit code verbatim —
    # the leader reads that code to know whether the pane teammate booted.
    seen: dict[str, object] = {}

    async def fake_main(**kwargs: object) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(_main_mod, "run_teammate_main", fake_main)
    rc = _main_mod.run_as_teammate(_teammate_ns(team_id="alpha", member="bob"))
    assert rc == 0
    assert seen["team_id"] == "alpha"
    assert seen["member_name"] == "bob"
    assert seen["agent_type"] == "general-purpose"


def test_run_as_teammate_keyboard_interrupt_returns_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ctrl-C during a teammate run is the conventional SIGINT exit (128+2).
    # The leader relies on 130 to distinguish a user abort from a crash.
    async def boom(**_kwargs: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(_main_mod, "run_teammate_main", boom)
    assert _main_mod.run_as_teammate(_teammate_ns()) == 130


# ---- subcommand dispatch in main() -----------------------------------------


def test_main_mcp_add_routes_post_dashdash_into_command_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `aura mcp add fs -- npx pkg`: tokens after `--` are the server's launch
    # command. argparse.REMAINDER can't coexist with --transport/--env, so
    # main() must hand the post-`--` tail to handle_mcp as args.command_args.
    captured: dict[str, object] = {}

    def fake_handle_mcp(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 7

    monkeypatch.setattr(_main_mod, "handle_mcp", fake_handle_mcp)
    monkeypatch.setattr(
        sys, "argv",
        ["aura", "mcp", "add", "fs", "--transport", "stdio", "--", "npx", "pkg"],
    )
    assert _main_mod.main() == 7
    ns = captured["args"]
    assert isinstance(ns, argparse.Namespace)
    assert ns.subcommand == "mcp"
    assert ns.mcp_action == "add"
    assert ns.command_args == ["npx", "pkg"]


def test_main_mcp_without_add_leaves_command_args_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `aura mcp list` has no `--` tail; main() must not synthesize one.
    # Only the `add` action consumes post-`--` tokens; other actions stay
    # untouched so handle_mcp sees a clean namespace.
    captured: dict[str, object] = {}

    def fake_handle_mcp(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(_main_mod, "handle_mcp", fake_handle_mcp)
    monkeypatch.setattr(sys, "argv", ["aura", "mcp", "list", "--", "stray"])
    assert _main_mod.main() == 0
    ns = captured["args"]
    assert isinstance(ns, argparse.Namespace)
    assert ns.mcp_action == "list"
    assert not hasattr(ns, "command_args")


def test_main_teammate_subcommand_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The `teammate` subcommand is the pane-teammate subprocess entrypoint;
    # main() must short-circuit to run_as_teammate (never build the REPL
    # agent) and propagate its exit code.
    captured: dict[str, object] = {}

    def fake_run_as_teammate(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 9

    monkeypatch.setattr(_main_mod, "run_as_teammate", fake_run_as_teammate)
    monkeypatch.setattr(
        sys, "argv",
        ["aura", "teammate", "--team-id", "T", "--member", "M",
         "--storage-root", "/tmp/x"],
    )
    assert _main_mod.main() == 9
    ns = captured["args"]
    assert isinstance(ns, argparse.Namespace)
    assert ns.team_id == "T"


# ---- config / ruleset load failures -> _fail_startup -----------------------


def test_main_config_load_failure_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # If load_config blows up at startup, main() must NOT crash with a raw
    # traceback — it routes through _fail_startup (exit 2, journaled). This
    # is the operator's "your config is broken" contract.
    project_dir = _write_min_layout(tmp_path, settings={})

    def boom_load() -> object:
        raise AuraError("bad config")

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["aura"])
    monkeypatch.setattr(_main_mod, "load_config", boom_load)
    assert _main_mod.main() == 2
    assert "AuraError: bad config" in capsys.readouterr().out


def test_main_invalid_ruleset_settings_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A malformed permissions.deny entry (non-string) makes store.load_*
    # raise AuraConfigError. main() must catch it and exit 2 via
    # _fail_startup rather than propagating a partially-built session.
    project_dir = _write_min_layout(tmp_path, settings={"deny": [123]})

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FAKE_API_KEY", "dummy")
    monkeypatch.setattr(sys, "argv", ["aura"])
    monkeypatch.setattr(_main_mod, "build_agent", lambda *a, **k: _FakeAgent())
    rc = _main_mod.main()
    out = capsys.readouterr().out
    assert rc == 2, f"expected exit 2; stdout={out!r}"
    assert "Error" in out


def test_main_build_agent_failure_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A failure while constructing the agent (e.g. an unknown provider in
    # the router) must surface as exit 2 through _fail_startup, not an
    # unhandled exception leaking the stack to the user's terminal.
    project_dir = _write_min_layout(tmp_path, settings={})
    _wire_main(monkeypatch, tmp_path, project_dir, argv=["aura"])

    def boom_build(*_a: object, **_k: object) -> _FakeAgent:
        raise ValueError("no such provider")

    monkeypatch.setattr(_main_mod, "build_agent", boom_build)
    assert _main_mod.main() == 2
    assert "startup error: no such provider" in capsys.readouterr().out


# ---- --log enablement -------------------------------------------------------


def test_main_log_flag_configures_journal_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `--log` must point the journal at ~/.aura/logs/events.jsonl and wrap
    # the hook chain with the event logger. Without this branch, opting into
    # logging would silently no-op and the audit trail would never persist.
    from aura.infrastructure.persistence import journal

    project_dir = _write_min_layout(tmp_path, settings={})
    configured: list[Path] = []
    monkeypatch.setattr(
        journal, "configure", lambda path: configured.append(path),
    )
    wrapped: list[bool] = []

    def fake_wrap(inner: HookChain) -> HookChain:
        wrapped.append(True)
        return inner

    _wire_main(monkeypatch, tmp_path, project_dir, argv=["aura", "--log"])
    monkeypatch.setattr(_main_mod, "wrap_with_event_logger", fake_wrap)
    assert _main_mod.main() == 0
    assert len(configured) == 1
    assert configured[0].name == "events.jsonl"
    assert wrapped == [True]


# ---- bypass banner + live-mode closure -------------------------------------


def test_main_bypass_flag_prints_banner_and_resolves_bypass_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # With --bypass-permissions and disable_bypass off, main() must enter
    # bypass mode AND surface the danger banner — silently disabling all
    # permission prompts without a visible warning would be a safety hole.
    project_dir = _write_min_layout(tmp_path, settings={"disable_bypass": False})
    captured = _wire_main(
        monkeypatch, tmp_path, project_dir, argv=["aura", "--bypass-permissions"],
    )
    assert _main_mod.main() == 0
    out = capsys.readouterr().out.lower()
    assert "bypass" in out
    live = captured["mode"]
    assert callable(live)


def test_live_mode_closure_falls_back_when_agent_mode_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The live-mode closure lets shift+tab toggles propagate to the
    # permission hook. If the agent ever reports a mode outside the known
    # set, the closure must fall back to the startup mode rather than leak
    # an invalid value into permission decisions.
    project_dir = _write_min_layout(tmp_path, settings={})

    class _OddModeAgent(_FakeAgent):
        mode = "wat-not-a-real-mode"

    captured = _wire_main(
        monkeypatch, tmp_path, project_dir, argv=["aura"],
        agent_factory=_OddModeAgent,
    )
    assert _main_mod.main() == 0
    live = captured["mode"]
    assert callable(live)
    assert live() == "default"  # startup mode, not the bogus agent value


@pytest.mark.parametrize("agent_mode", ["default", "bypass", "plan", "accept_edits"])
def test_live_mode_closure_reflects_known_agent_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_mode: str,
) -> None:
    # Every member of the known Mode set must round-trip through the closure
    # unchanged — that's what makes a runtime shift+tab toggle authoritative
    # over the startup mode for the next permission check.
    project_dir = _write_min_layout(tmp_path, settings={})

    class _ModedAgent(_FakeAgent):
        mode = agent_mode

    captured = _wire_main(
        monkeypatch, tmp_path, project_dir, argv=["aura"],
        agent_factory=_ModedAgent,
    )
    assert _main_mod.main() == 0
    live = captured["mode"]
    assert callable(live)
    assert live() == agent_mode


# ---- _entry resilience: aconnect / watcher failures must not crash ---------


def test_main_entry_swallows_aconnect_and_watcher_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # MCP connect and the file watcher are best-effort: a failure in either
    # must be logged and tolerated, never abort the REPL. A user with a
    # broken MCP server or an unwatchable cwd still gets a working session.
    project_dir = _write_min_layout(tmp_path, settings={})

    class _FlakyAgent(_FakeAgent):
        async def aconnect(self) -> None:
            raise RuntimeError("mcp down")

    class _FlakyWatcher(_FakeWatcher):
        async def start(self) -> None:
            raise OSError("cannot watch")

    repl_ran: list[bool] = []

    async def repl_marker(*_a: object, **_k: object) -> None:
        repl_ran.append(True)

    _wire_main(
        monkeypatch, tmp_path, project_dir, argv=["aura"],
        agent_factory=_FlakyAgent, repl=repl_marker,
    )
    monkeypatch.setattr(_main_mod, "FileWatcher", _FlakyWatcher)
    assert _main_mod.main() == 0
    assert repl_ran == [True]  # REPL still ran despite both subsystems failing
    assert "mcp connect error" in capsys.readouterr().out


def test_main_keyboard_interrupt_in_repl_returns_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ctrl-C at the REPL is a clean user-initiated shutdown: main() must
    # return 130 (SIGINT convention), not a traceback, and still run its
    # finally-block cleanup (agent.close + shutdown journal event).
    from aura.infrastructure.persistence import journal

    project_dir = _write_min_layout(tmp_path, settings={})
    events: list[str] = []
    monkeypatch.setattr(
        journal, "write",
        lambda event, /, **_f: events.append(event),
    )

    async def repl_sigint(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    _wire_main(
        monkeypatch, tmp_path, project_dir, argv=["aura"], repl=repl_sigint,
    )
    assert _main_mod.main() == 130
    assert "shutdown_sigint" in events
    assert "shutdown" in events  # finally-block cleanup still fired
