"""Aura: a general-purpose Python agent with an explicit async loop."""

from aura.application.agent_context import AgentContext
from aura.application.run_agent import run_agent
from aura.application.session import AgentSession
from aura.config.loader import load_config
from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.agent import Agent, build_agent
from aura.domain.agent_definition import AgentDefinition
from aura.domain.errors import AuraError

__version__ = "0.11.0"

__all__ = [
    "Agent",
    "AgentContext",
    "AgentDefinition",
    "AgentSession",
    "AuraConfig",
    "AuraConfigError",
    "AuraError",
    "__version__",
    "build_agent",
    "load_config",
    "run_agent",
]
