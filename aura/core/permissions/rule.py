"""Permission rule: ``Rule(tool, content)`` — parsed from strings like
``"bash"`` (tool-wide) or ``"bash(npm test)"`` (pattern).

Spec: ``docs/specs/2026-04-19-aura-permission.md`` §3.1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from langchain_core.tools import BaseTool

from aura.errors import AuraError


class InvalidRuleError(AuraError):
    """Raised when a rule string cannot be parsed."""


_RuleMatcher = Callable[[dict[str, Any], str], bool]


@dataclass(frozen=True)
class Rule:
    tool: str
    content: str | None

    def matches(self, tool_name: str, args: dict[str, Any], tool: BaseTool) -> bool:
        """True iff this rule covers a call of ``tool_name`` with ``args``.

        Resolution:
        1. Name mismatch → False.
        2. Tool-wide rule (``content is None``) → True, regardless of args.
        3. Pattern rule → delegate to the tool's ``rule_matcher`` in metadata;
           absent matcher → False (conservative: a tool that never declared
           how to match arg patterns cannot be allowed by a pattern rule).
        """
        if tool_name != self.tool:
            return False
        if self.content is None:
            return True
        matcher = (tool.metadata or {}).get("rule_matcher")
        if matcher is None:
            return False
        return cast(_RuleMatcher, matcher)(args, self.content)

    def to_string(self) -> str:
        if self.content is None:
            return self.tool
        # Order matters: escape `\` before `(` / `)`, otherwise the escape
        # sequences themselves get double-escaped.
        escaped = (
            self.content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        return f"{self.tool}({escaped})"

    @classmethod
    def parse(cls, raw: str) -> Rule:
        if not raw:
            raise InvalidRuleError("rule", "empty string is not a valid rule")

        # Tool-wide: no parens at all.
        if "(" not in raw:
            return cls(tool=raw, content=None)

        # Pattern: `tool(content)`. Tool name is everything before the first
        # `(`; content follows, terminated by the first unescaped `)`; nothing
        # may trail the close paren.
        open_idx = raw.index("(")
        tool = raw[:open_idx]
        if not tool:
            raise InvalidRuleError("rule", f"empty tool name in {raw!r}")

        chars: list[str] = []
        i = open_idx + 1
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                chars.append(raw[i + 1])
                i += 2
                continue
            if ch == ")":
                if i != len(raw) - 1:
                    trailing = raw[i + 1 :]
                    raise InvalidRuleError(
                        "rule",
                        f"trailing characters after close paren: {trailing!r}",
                    )
                return cls(tool=tool, content="".join(chars))
            chars.append(ch)
            i += 1

        raise InvalidRuleError("rule", f"unclosed paren in {raw!r}")
