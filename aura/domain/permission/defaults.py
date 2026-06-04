"""Built-in default allow-rules merged with user rules at startup."""

from __future__ import annotations

from aura.domain.permission.rule import Rule

DEFAULT_ALLOW_RULES: tuple[Rule, ...] = (
    Rule(tool="read_file", content=None),
    Rule(tool="grep", content=None),
    Rule(tool="glob", content=None),
    Rule(tool="ask_user_question", content=None),
)

__all__ = ["DEFAULT_ALLOW_RULES"]
