"""Permission rule parsed from strings like ``bash`` or ``bash(npm test)``.

Match resolution: tool-name (exact, or fnmatch when * in self.tool) then
content (None = tool-wide match; pattern delegates to the tool's
rule_matcher metadata, absent matcher = no match).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal, cast

from langchain_core.tools import BaseTool

from aura.domain.errors import AuraError
from aura.domain.tool_meta_access import meta_dict


class InvalidRuleError(AuraError):
    pass


_RuleMatcher = Callable[[dict[str, Any], str], bool]

RuleKind = Literal["allow", "deny", "ask"]


@dataclass(frozen=True)
class Rule:
    tool: str
    content: str | None
    kind: RuleKind = field(default="allow")

    def matches(self, tool_name: str, args: dict[str, Any], tool: BaseTool) -> bool:
        if "*" in self.tool:
            if not fnmatchcase(tool_name, self.tool):
                return False
        elif tool_name != self.tool:
            return False
        if self.content is None:
            return True
        matcher = meta_dict(tool).get("rule_matcher")
        if matcher is None:
            return False
        return cast(_RuleMatcher, matcher)(args, self.content)

    def to_string(self) -> str:
        if self.content is None:
            return self.tool
        # Escape backslash before parens to keep the round-trip invertible.
        escaped = (
            self.content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        return f"{self.tool}({escaped})"

    @classmethod
    def parse(cls, raw: str, *, kind: RuleKind = "allow") -> Rule:
        if not raw:
            raise InvalidRuleError("rule", "empty string is not a valid rule")

        if "(" not in raw:
            return cls(tool=raw, content=None, kind=kind)

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
                return cls(tool=tool, content="".join(chars), kind=kind)
            chars.append(ch)
            i += 1

        raise InvalidRuleError("rule", f"unclosed paren in {raw!r}")
