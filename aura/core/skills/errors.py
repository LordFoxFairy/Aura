"""Shared error formatting for skill invocation paths.

Both the slash-command surface (:class:`SkillCommand`) and the model-facing
tool surface (:class:`SkillTool`) enforce the same "declared arguments
must all be provided" contract. The text of the resulting error must match
byte-for-byte so users / models who've seen one path recognise the error on
the other — drift between the two was the v0.11 bug this helper closes.
"""

from __future__ import annotations


def format_missing_args_error(
    name: str, declared: tuple[str, ...], provided: int,
) -> str:
    """Return the canonical missing-required-args message.

    ``name`` is the skill name (no leading slash), ``declared`` is the full
    tuple of argument names from the skill's frontmatter, ``provided`` is
    how many positional values the caller actually passed. Callers are
    responsible for deciding whether to raise (tool path → ToolError) or
    render (slash path → CommandResult).
    """
    missing = list(declared[provided:])
    return (
        f"skill {name!r} requires arguments {list(declared)}; "
        f"missing: {missing}"
    )
