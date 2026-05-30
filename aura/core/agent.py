"""Transitional entry: ``Agent`` alias over ``AgentSession`` + ``build_agent``.

``Agent`` is a stage-bridge alias (call sites still import it); removed in P4
once ``build_session`` unifies construction.
"""

from aura.application.session import AgentSession, build_agent

Agent = AgentSession

__all__ = ["Agent", "AgentSession", "build_agent"]
