"""Aura — a general-purpose Python agent with an explicit async loop."""

# errors に依存がないため最初にロード — config.schema が aura.core.errors を要求する際に
# sys.modules に既に存在し、循環 import を回避できる。
from aura.core.errors import AuraError
from aura.config.loader import load_config
from aura.config.schema import AuraConfig, AuraConfigError
from aura.core.agent import Agent, build_agent

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
