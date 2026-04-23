"""Subagent auto-deny asker — parity with claude-code ``runAgent.ts:440-451``.

When a subagent's permission hook would otherwise prompt the user
(``_decide`` hits the ``ask`` branch), there's nobody to ask:
subagents have no UI and no attention stream. The canonical claude-code
contract is to flip ``shouldAvoidPermissionPrompts: true`` on the child
:class:`ToolPermissionContext` so that any would-be prompt silently
becomes a deny.

Aura replicates that by handing the subagent's permission hook an
:class:`AskerResponse`-producing callable that always returns
``choice="deny"``. The ``feedback`` slot carries a fixed marker string
(``"subagent_auto_deny"``) so the hook's ``_deny_message`` composer
stamps the provenance onto the model-facing error — otherwise the LLM
would see a generic "denied: user" and have no way to tell it was
ambient policy, not a human pressing N.

Zero I/O: the asker returns synchronously from an async coroutine,
burning no event-loop slot. Safe to share one instance across every
subagent / tool call.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from aura.core.hooks.permission import AskerResponse
from aura.core.permissions.rule import Rule

# Exposed so hook + test code can both reference the same constant
# rather than re-typing the literal.
SUBAGENT_AUTO_DENY_FEEDBACK = "subagent_auto_deny"


class SubagentAutoDenyAsker:
    """PermissionAsker that denies every call without asking.

    Protocol-compatible with :class:`aura.core.hooks.permission
    .PermissionAsker` — keyword-only ``__call__`` taking
    ``tool`` / ``args`` / ``rule_hint`` and returning an
    :class:`AskerResponse`.
    """

    async def __call__(
        self,
        *,
        tool: BaseTool,  # noqa: ARG002 — protocol compliance
        args: dict[str, Any],  # noqa: ARG002
        rule_hint: Rule,  # noqa: ARG002
    ) -> AskerResponse:
        return AskerResponse(
            choice="deny",
            scope="session",
            rule=None,
            feedback=SUBAGENT_AUTO_DENY_FEEDBACK,
        )
