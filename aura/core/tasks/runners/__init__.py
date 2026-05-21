"""Subagent runner topologies — claude-code Task hierarchy parity.

Three runner shapes, mirroring claude-code's ``tasks/`` directory:

- :class:`LocalAgentTask` — in-process child Agent, fire-and-forget,
  single prompt → single terminal outcome. The default :func:`run_task`
  path uses this.
- :class:`InProcessTeammateTask` — long-lived in-process teammate
  joined to a team mailbox. Adapts
  :func:`aura.core.teams.runtime.run_teammate`.
- :class:`RemoteAgentTask` — subprocess teammate launched via
  ``python -m cli.teammate_entrypoint`` (or an equivalent entrypoint
  module). The pane backend already uses this in spirit; the runner
  formalises it as a first-class topology.

All three classes share a ``start() / wait_for_terminal() / abort()``
contract so callers can dispatch uniformly without branching on
backend type.
"""

from aura.core.tasks.runners.in_process import InProcessTeammateTask
from aura.core.tasks.runners.local_agent import LocalAgentTask
from aura.core.tasks.runners.remote import (
    DEFAULT_ENTRYPOINT_MODULE,
    RemoteAgentTask,
)

__all__ = [
    "DEFAULT_ENTRYPOINT_MODULE",
    "InProcessTeammateTask",
    "LocalAgentTask",
    "RemoteAgentTask",
]
