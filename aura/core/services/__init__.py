"""Long-lived background services that orbit the Agent loop.

A "service" here is a non-LangChain, non-tool background coroutine
that runs alongside the main turn cycle and produces side effects on
the parent agent's observable state (TasksStore progress, journal,
…).  Today the only service is :class:`AgentSummarizer` (Round 7QS —
periodic LLM-driven digest of a running subagent) but the package
gives later additions (a remote-agent poller, a scheduled-task
fanout) a clear home that doesn't pollute ``aura.core.tasks`` or
``aura.tools``.
"""

from __future__ import annotations

from aura.core.services.agent_summary import AgentSummarizer, run_summary_loop

__all__ = ["AgentSummarizer", "run_summary_loop"]
