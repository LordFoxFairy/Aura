"""Subagent runner — single LocalAgentTask topology.

``task_create`` always dispatches an in-process child Agent: one prompt,
one terminal outcome, fire-and-forget. Teams use their own runtime in
:mod:`aura.application.teams.runtime`, not a runner here.
"""

from aura.application.tasks.runners.local_agent import LocalAgentTask

__all__ = ["LocalAgentTask"]
