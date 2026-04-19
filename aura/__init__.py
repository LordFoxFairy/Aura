"""Aura: a general-purpose Python agent with an explicit async loop."""

from aura.config.loader import load_config
from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.agent import Agent, build_agent
from aura.errors import AuraError

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AuraConfig",
    "AuraConfigError",
    "AuraError",
    "__version__",
    "build_agent",
    "load_config",
]
