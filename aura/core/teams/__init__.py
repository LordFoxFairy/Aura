"""Aura Teams — long-lived, mailbox-coupled multi-agent sessions.

Phase A. One team per leader session, in-process teammates only,
file-rooted at ``<storage_root>/teams/<team_id>/``. Communication is
JSONL mailbox + ``.seen`` cursor sidecar; teammates inherit the leader's
permission rules bit-for-bit (mirror :func:`SubagentFactory.spawn`).

Public surface:

- :class:`TeamRecord` / :class:`TeammateMember` / :class:`TeamMessage` —
  on-disk types.
- :class:`Mailbox` — append + read-unseen + ack.
- :class:`TeamManager` — owns lifecycle (create / add / remove / send).
- :func:`run_teammate` — long-lived loop driving one teammate's Agent.
"""

from aura.core.teams.mailbox import Mailbox
from aura.core.teams.manager import TeamManager
from aura.core.teams.runtime import run_teammate
from aura.core.teams.types import TeammateMember, TeamMessage, TeamRecord

__all__ = [
    "Mailbox",
    "TeamManager",
    "TeamMessage",
    "TeamRecord",
    "TeammateMember",
    "run_teammate",
]
