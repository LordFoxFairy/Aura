"""Core agent loop, state, registry, hooks, and built-in hook factories."""

from aura.core.agent import Agent, build_agent
from aura.core.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)
from aura.core.hooks import (
    HookChain,
    PostModelHook,
    PostToolHook,
    PreModelHook,
    PreToolHook,
)
from aura.core.hooks.budget import (
    MaxTurnsExceeded,
    default_hooks,
    make_max_turns_hook,
    make_size_budget_hook,
    make_usage_tracking_hook,
)
from aura.core.hooks.permission import PermissionAsker, PermissionSession, make_permission_hook
from aura.core.llm import (
    MissingCredentialError,
    MissingProviderDependencyError,
    UnknownModelSpecError,
)
from aura.core.loop import AgentLoop, ToolStep
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.core.state import LoopState
from aura.errors import AuraError

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentLoop",
    "AssistantDelta",
    "AuraError",
    "Final",
    "HookChain",
    "LoopState",
    "MaxTurnsExceeded",
    "MissingCredentialError",
    "MissingProviderDependencyError",
    "PermissionAsker",
    "PermissionSession",
    "PostModelHook",
    "PostToolHook",
    "PreModelHook",
    "PreToolHook",
    "SessionStorage",
    "ToolCallCompleted",
    "ToolCallStarted",
    "ToolRegistry",
    "ToolStep",
    "UnknownModelSpecError",
    "build_agent",
    "default_hooks",
    "journal",
    "make_max_turns_hook",
    "make_permission_hook",
    "make_size_budget_hook",
    "make_usage_tracking_hook",
]
