"""Smoke test: public API surface stays reachable from package roots."""

from __future__ import annotations


def test_aura_top_level_exports() -> None:
    from aura import (  # noqa: F401
        Agent,
        AuraConfig,
        AuraConfigError,
        AuraError,
        build_agent,
        load_config,
    )

    assert Agent.__name__ == "Agent"
    assert AuraConfig.__name__ == "AuraConfig"
    assert AuraConfigError.__name__ == "AuraConfigError"
    assert AuraError.__name__ == "AuraError"
    assert issubclass(AuraConfigError, AuraError)
    assert callable(build_agent)
    assert callable(load_config)


def test_aura_core_exports() -> None:
    from aura.core import (  # noqa: F401
        Agent,
        AgentEvent,
        AgentLoop,
        AssistantDelta,
        AuraError,
        Final,
        HookChain,
        LoopState,
        ModelFactory,
        PostModelHook,
        PostToolHook,
        PreModelHook,
        PreToolHook,
        SessionStorage,
        ToolCallCompleted,
        ToolCallStarted,
        ToolRegistry,
        ToolStep,
        make_size_budget_hook,
        make_usage_tracking_hook,
    )

    assert callable(make_size_budget_hook)
    assert callable(make_usage_tracking_hook)
    assert AgentLoop.__name__ == "AgentLoop"


def test_aura_tools_exports() -> None:
    from aura.tools import (  # noqa: F401
        AuraTool,
        ToolResult,
        bash,
        build_tool,
        edit_file,
        glob,
        grep,
        read_file,
        web_fetch,
        write_file,
    )

    assert callable(build_tool)
    assert bash.name == "bash"
    assert edit_file.name == "edit_file"
    assert glob.name == "glob"
    assert grep.name == "grep"
    assert read_file.name == "read_file"
    assert web_fetch.name == "web_fetch"
    assert write_file.name == "write_file"


def test_aura_config_exports() -> None:
    from aura.config import (  # noqa: F401
        AuraConfig,
        AuraConfigError,
        LogConfig,
        ProviderConfig,
        StorageConfig,
        ToolsConfig,
        UIConfig,
        load_config,
    )

    assert AuraConfig.__name__ == "AuraConfig"
    assert LogConfig.__name__ == "LogConfig"
    assert callable(load_config)
