"""Aura Teams — long-lived, mailbox-coupled multi-agent sessions.

Application layer: orchestration over the on-disk
:class:`~aura.domain.team.TeamRecord` and the
:func:`~aura.domain.team_memory.redact_secrets` policy. One team per
leader session, in-process teammates by default, file-rooted at
``<storage_root>/teams/<team_id>/``. Communication is JSONL mailbox +
``.seen`` cursor sidecar; teammates inherit the leader's permission
rules.
"""
