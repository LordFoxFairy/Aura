"""Aura-native shape adapters over ``langchain-mcp-adapters`` objects."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from aura.application.commands.types import CommandResult, CommandSource
from aura.schemas.tool import ToolMetadata

if TYPE_CHECKING:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from aura.core.agent import Agent


_MCP_PREFIX = "mcp__"
_MAX_MCP_RESULT_CHARS = 30_000
_MCP_DESCRIPTION_CAP = 2048
_DEFAULT_OP_TIMEOUT_SEC = 30.0


def _args_preview(args: dict[str, Any]) -> str:
    if not args:
        return "args: (none)"
    # Keys only; values may be large or sensitive.
    return "args: " + ", ".join(sorted(args.keys()))


def _cap_description(text: str) -> tuple[str, bool, int]:
    original_len = len(text)
    if original_len <= _MCP_DESCRIPTION_CAP:
        return text, False, original_len
    head = text[: _MCP_DESCRIPTION_CAP - 64]
    marker = f"\n[truncated; original was {original_len} chars]"
    return head + marker, True, original_len


def _read_annotation_hints(tool: BaseTool) -> dict[str, bool | None]:
    hints: dict[str, bool | None] = {
        "readOnlyHint": None,
        "destructiveHint": None,
        "openWorldHint": None,
    }
    md = getattr(tool, "metadata", None)
    if isinstance(md, dict):
        for key in hints:
            val = md.get(key)
            if isinstance(val, bool):
                hints[key] = val
    ann = getattr(tool, "annotations", None)
    if ann is not None:
        for key in hints:
            val = getattr(ann, key, None)
            if isinstance(val, bool):
                hints[key] = val
    return hints


def add_aura_metadata(tool: BaseTool, *, server_name: str) -> BaseTool:
    """Namespace *tool*'s name and attach Aura's typed metadata in-place.

    Defaults conservatively (destructive + non-concurrency-safe) when the
    server omits :class:`ToolAnnotations` hints; readOnlyHint /
    destructiveHint / openWorldHint flip the defaults when declared.
    Returns the same object for call-site convenience.
    """
    hints = _read_annotation_hints(tool)
    if not tool.name.startswith(_MCP_PREFIX):
        tool.name = f"{_MCP_PREFIX}{server_name}__{tool.name}"
    capped_desc, truncated, original_len = _cap_description(
        tool.description or ""
    )
    if truncated:
        tool.description = capped_desc
        try:
            from aura.core import journal  # noqa: PLC0415  # deferred import is intentional
            journal.write(
                "mcp_description_truncated",
                tool_name=tool.name,
                server=server_name,
                original_len=original_len,
                truncated_len=len(capped_desc),
            )
        except Exception:  # noqa: BLE001  # logging path must never crash caller
            pass
    is_read_only = hints["readOnlyHint"] is True
    is_destructive = not (is_read_only or hints["destructiveHint"] is False)
    is_concurrency_safe = is_read_only or hints["openWorldHint"] is False
    aura_meta = ToolMetadata(
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        is_concurrency_safe=is_concurrency_safe,
        rule_matcher=None,
        args_preview=_args_preview,
        timeout_sec=None,
        max_result_size_chars=_MAX_MCP_RESULT_CHARS,
    )
    object.__setattr__(tool, "aura_metadata", aura_meta)
    tool.metadata = None
    return tool


def expand_env_vars(
    text: str,
    *,
    _missing_log: list[str] | None = None,
) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references in *text*.

    Invariant: output is NOT recursively re-expanded — a value containing
    ``${...}`` cannot read another env variable on use. Empty-string env
    values count as missing (shell semantics).
    """
    import os
    import re

    if "${" not in text:
        return text

    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        val = os.environ.get(name, "")
        if val:
            return val
        if default is not None:
            return default
        if _missing_log is not None and name not in _missing_log:
            _missing_log.append(name)
        return ""

    return pattern.sub(_sub, text)


class _MCPPromptCommand:
    """Slash-command that fetches an MCP prompt body on demand and prints it.

    Positional tokens at invocation are zipped against the prompt's declared
    ``arguments`` and forwarded as ``arguments={}``; missing required names
    raise a user-visible error, extras are dropped.
    """

    source: CommandSource = "mcp"
    allowed_tools: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        server_name: str,
        prompt_name: str,
        description: str,
        client: MultiServerMCPClient,
        arg_names: tuple[str, ...] = (),
        required_args: frozenset[str] = frozenset(),
        op_timeout_sec: float = _DEFAULT_OP_TIMEOUT_SEC,
    ) -> None:
        self._server = server_name
        self._prompt = prompt_name
        self._client = client
        self._op_timeout_sec = op_timeout_sec
        self.name = f"/{server_name}__{prompt_name}"
        self.description = description
        self.arg_names: tuple[str, ...] = arg_names
        self.required_args: frozenset[str] = required_args
        if arg_names:
            parts = [
                f"<{n}>" if n in required_args else f"[{n}]"
                for n in arg_names
            ]
            self.argument_hint: str | None = " ".join(parts)
        else:
            self.argument_hint = None

    def _build_arguments(
        self, arg: str
    ) -> tuple[dict[str, str] | None, str | None]:
        """Zip positional tokens against ``arg_names``; surface missing-required as error."""
        tokens = arg.split() if arg else []
        provided: dict[str, str] = {}
        for i, name in enumerate(self.arg_names):
            if i < len(tokens):
                provided[name] = tokens[i]
        missing = [n for n in self.arg_names if n in self.required_args and n not in provided]
        if missing:
            return None, (
                f"mcp prompt {self._server}:{self._prompt} "
                f"missing required argument(s): {', '.join(missing)}"
            )
        return provided, None

    async def handle(self, arg: str, agent: Agent) -> CommandResult:
        from aura.core import journal

        arguments, error = self._build_arguments(arg)
        if arguments is None:
            # error is non-None whenever arguments is None per _build_arguments.
            assert error is not None
            journal.write(
                "mcp_prompt_missing_args",
                server=self._server,
                prompt=self._prompt,
                arg_names=list(self.arg_names),
                required=sorted(self.required_args),
            )
            return CommandResult(handled=True, kind="print", text=error)

        try:
            messages = await asyncio.wait_for(
                self._client.get_prompt(
                    self._server,
                    self._prompt,
                    arguments=arguments,
                ),
                timeout=self._op_timeout_sec,
            )
        except TimeoutError:
            journal.write(
                "mcp_prompt_fetch_timeout",
                server=self._server,
                prompt=self._prompt,
                timeout_sec=self._op_timeout_sec,
            )
            return CommandResult(
                handled=True,
                kind="print",
                text=(
                    f"mcp prompt fetch timed out after "
                    f"{self._op_timeout_sec}s "
                    f"(server {self._server!r}, prompt {self._prompt!r})"
                ),
            )
        except Exception as exc:  # noqa: BLE001  # surface server-side failure to user, don't propagate
            journal.write(
                "mcp_prompt_fetch_failed",
                server=self._server,
                prompt=self._prompt,
                error=f"{type(exc).__name__}: {exc}",
            )
            return CommandResult(
                handled=True,
                kind="print",
                text=f"mcp prompt fetch failed: {exc}",
            )

        text = "\n".join(str(m.content) for m in messages)
        journal.write(
            "mcp_prompt_invoked",
            server=self._server,
            prompt=self._prompt,
            arg_count=len(arguments),
        )
        return CommandResult(handled=True, kind="print", text=text)


def make_mcp_command(
    *,
    server_name: str,
    prompt_name: str,
    prompt_description: str,
    client: MultiServerMCPClient,
    prompt_arguments: list[Any] | None = None,
    op_timeout_sec: float = _DEFAULT_OP_TIMEOUT_SEC,
) -> _MCPPromptCommand:
    """Build a slash :class:`Command` that fetches and renders an MCP prompt."""
    arg_names: tuple[str, ...] = ()
    required_args: frozenset[str] = frozenset()
    if prompt_arguments:
        names: list[str] = []
        required: set[str] = set()
        for pa in prompt_arguments:
            name = getattr(pa, "name", None)
            if not isinstance(name, str) or not name:
                continue
            names.append(name)
            if getattr(pa, "required", False) is True:
                required.add(name)
        arg_names = tuple(names)
        required_args = frozenset(required)
    return _MCPPromptCommand(
        server_name=server_name,
        prompt_name=prompt_name,
        description=prompt_description,
        client=client,
        arg_names=arg_names,
        required_args=required_args,
        op_timeout_sec=op_timeout_sec,
    )


def normalize_resource_contents(contents: Any) -> dict[str, Any]:
    """Flatten an MCP ``ResourceContents`` object into a JSON-safe dict.

    Invariant: ``BlobResourceContents.blob`` is reported by decoded byte
    size only — base64 payloads are NOT echoed back to the LLM.
    """
    import base64

    uri = getattr(contents, "uri", None)
    mime = getattr(contents, "mimeType", None)
    text = getattr(contents, "text", None)
    if isinstance(text, str):
        return {
            "type": "text",
            "uri": None if uri is None else str(uri),
            "mime": mime,
            "text": text,
        }
    blob = getattr(contents, "blob", None)
    if isinstance(blob, str):
        try:
            size = len(base64.b64decode(blob, validate=False))
        except (ValueError, TypeError):
            size = len(blob)
        return {
            "type": "blob",
            "uri": None if uri is None else str(uri),
            "mime": mime,
            "size": size,
        }
    return {
        "type": "unknown",
        "uri": None if uri is None else str(uri),
        "mime": mime,
        "repr": repr(contents),
    }
