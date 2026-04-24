"""Core agent loop, state, registry, hooks, and built-in hook factories."""

from aura.core.agent import Agent, build_agent
from aura.core.hooks import (
    PRE_TOOL_PASSTHROUGH,
    HookChain,
    PostModelHook,
    PostSessionHook,
    PostSubagentHook,
    PostToolHook,
    PreCompactHook,
    PreModelHook,
    PreSessionHook,
    PreToolHook,
    PreToolOutcome,
    PreUserPromptHook,
)
from aura.core.hooks.budget import (
    default_hooks,
    make_size_budget_hook,
    make_usage_tracking_hook,
)
from aura.core.hooks.permission import (
    AskerResponse,
    PermissionAsker,
    make_permission_hook,
)
from aura.core.llm import (
    MissingCredentialError,
    MissingProviderDependencyError,
    UnknownModelSpecError,
)
from aura.core.loop import AgentLoop, ToolStep
from aura.core.permissions.denials import PermissionDenial
from aura.core.persistence import journal
from aura.core.persistence.storage import SessionStorage
from aura.core.registry import ToolRegistry
from aura.errors import AuraError
from aura.schemas.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)
from aura.schemas.state import LoopState

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentLoop",
    "AskerResponse",
    "AssistantDelta",
    "AuraError",
    "Final",
    "HookChain",
    "LoopState",
    "MissingCredentialError",
    "MissingProviderDependencyError",
    "PermissionAsker",
    "PermissionDenial",
    "PostModelHook",
    "PostSessionHook",
    "PostSubagentHook",
    "PostToolHook",
    "PreCompactHook",
    "PreModelHook",
    "PreSessionHook",
    "PRE_TOOL_PASSTHROUGH",
    "PreToolHook",
    "PreToolOutcome",
    "PreUserPromptHook",
    "SessionStorage",
    "ToolCallCompleted",
    "ToolCallStarted",
    "ToolRegistry",
    "ToolStep",
    "UnknownModelSpecError",
    "build_agent",
    "default_hooks",
    "journal",
    "make_permission_hook",
    "make_size_budget_hook",
    "make_usage_tracking_hook",
]
