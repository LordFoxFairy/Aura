"""Core agent loop, state, registry, hooks, and built-in hook factories."""

from aura.application.hooks import (
    HookChain,
    PostModelHook,
    PostToolHook,
    PreModelHook,
    PreToolHook,
)
from aura.application.hooks.budget import (
    default_hooks,
    make_size_budget_hook,
    make_usage_tracking_hook,
)
from aura.application.hooks.permission import make_permission_hook
from aura.application.loop import AgentLoop, ToolStep
from aura.application.loop_state import LoopState
from aura.application.permission.asker import AskerResponse, PermissionAsker
from aura.core.agent import Agent, AgentSession, build_agent
from aura.domain.errors import AuraError
from aura.domain.events import (
    AgentEvent,
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallStarted,
)
from aura.domain.permission.denials import PermissionDenial
from aura.domain.tool_registry import ToolRegistry
from aura.infrastructure.llm import (
    MissingCredentialError,
    MissingProviderDependencyError,
    UnknownModelSpecError,
)
from aura.infrastructure.persistence import journal
from aura.infrastructure.persistence.storage import SessionStorage

__all__ = [
    "Agent",
    "AgentSession",
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
    "make_permission_hook",
    "make_size_budget_hook",
    "make_usage_tracking_hook",
]
