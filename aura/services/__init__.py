"""Cross-cutting service modules (Phase E parity layer).

This is a deliberately small package: each module here is a leaf
service consumed by one or more code paths in :mod:`aura.core` /
:mod:`aura.tools`. Services do NOT depend on each other; if two
services need to coordinate they go through the store / journal /
storage primitives, not through a shared in-process state.

Distinct from :mod:`aura.core.services` — that package holds
infrastructure-level services that are tightly coupled to the agent
loop (e.g. the periodic AgentSummarizer that ticks a cheap-model
digest during a running subagent). :mod:`aura.services` is for
synchronous, callable-from-anywhere helpers that have no event-loop
involvement of their own.
"""

from aura.services.agent_summary import summarize_subagent_run

__all__ = ["summarize_subagent_run"]
