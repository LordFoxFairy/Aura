"""Aura's thin wrapper around ``langchain-mcp-adapters`` BaseTool / prompt objects.

Three public functions — no classes — because each one maps a plain library
object onto an Aura-native shape:

- :func:`add_aura_metadata` mutates a :class:`BaseTool` returned by
  :class:`MultiServerMCPClient` in place to attach Aura's capability flags
  + namespace its ``.name``. MCP servers do not self-declare destructive /
  concurrency-safe / size-bound semantics, so we default conservatively
  (destructive = True, concurrency-safe = False, size cap = 30k chars).
  The permission layer therefore prompts by default until the user adds
  a rule for the tool.

- :func:`make_mcp_command` wraps an MCP *prompt* as an Aura
  :class:`Command` so it can be registered into :class:`CommandRegistry`
  alongside built-ins and skill commands. ``source="mcp"`` lets ``/help``
  group it separately.

- :func:`normalize_resource_contents` converts MCP's
  ``TextResourceContents`` / ``BlobResourceContents`` pydantic objects
  into plain JSON-serialisable dicts. The ``mcp_read_resource`` tool
  returns these through LangChain's tool-result channel (which stringifies
  via ``json.dumps``); shipping the raw pydantic objects fails because
  ``ResourceContents.uri`` is an ``AnyUrl`` with no default JSON encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from aura.core.commands.types import CommandResult, CommandSource
from aura.schemas.tool import tool_metadata

if TYPE_CHECKING:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    from aura.core.agent import Agent


_MCP_PREFIX = "mcp__"
_MAX_MCP_RESULT_CHARS = 30_000


def _args_preview(args: dict[str, Any]) -> str:
    """Generic one-line preview for MCP tool calls.

    MCP tools expose arbitrary JSON-Schema arg shapes, so we can't build a
    tool-specific preview (like bash(command=...)). The CLI permission
    prompt falls back to this when rendering an MCP tool call.
    """
    if not args:
        return "args: (none)"
    # Show keys only — values may be large or sensitive (auth tokens etc).
    return "args: " + ", ".join(sorted(args.keys()))


def add_aura_metadata(tool: BaseTool, *, server_name: str) -> BaseTool:
    """Mutate *tool* in-place: namespace the name and attach Aura metadata.

    Name is normalised to ``mcp__<server_name>__<original>``. If the library
    has already applied a prefix (tools coming back already namespaced), we
    detect the ``mcp__`` prefix and leave the name alone.

    Returns the same object for call-site convenience.
    """
    # StructuredTool.name / metadata are plain pydantic fields — mutating
    # them is supported by langchain-core. No reflection needed.
    if not tool.name.startswith(_MCP_PREFIX):
        tool.name = f"{_MCP_PREFIX}{server_name}__{tool.name}"
    tool.metadata = tool_metadata(
        is_read_only=False,
        is_destructive=True,
        is_concurrency_safe=False,
        max_result_size_chars=_MAX_MCP_RESULT_CHARS,
        rule_matcher=None,
        args_preview=_args_preview,
    )
    return tool


class _MCPPromptCommand:
    """Command that pulls an MCP prompt body on demand and prints it.

    We don't inject the body into Context on invocation — this keeps parity
    with SkillCommand's surface (``handled=True, kind="print"``). A later
    release can route prompt bodies through Context if MCP prompts become
    multi-turn templates instead of one-shot messages.

    When the underlying MCP prompt declares ``arguments`` (see
    ``mcp.types.PromptArgument``), invocation-time positional tokens
    (``/server__prompt alpha beta``) are parsed and forwarded as a named
    ``arguments={}`` dict — matching claude-code's behaviour in
    ``src/services/mcp/client.ts`` (``zipObject(argNames, argsArray)``
    around lines 2055 / 2077). Missing *required* args produce a
    user-visible error instead of silently sending ``None``; extra
    positionals are dropped silently (parity with lodash ``zipObject``).
    """

    source: CommandSource = "mcp"
    # MCP prompts do not self-declare tool-gating today; we keep the
    # shape symmetric with SkillCommand / built-ins (empty tuple = "no
    # explicit gating") so ``/help`` and future inspectors can treat the
    # Command surface uniformly across sources.
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
    ) -> None:
        self._server = server_name
        self._prompt = prompt_name
        self._client = client
        self.name = f"/{server_name}__{prompt_name}"
        self.description = description
        # Argument schema captured at registration time (see
        # ``make_mcp_command``). Preserved as an ordered tuple so positional
        # tokens at invocation zip deterministically against the names.
        self.arg_names: tuple[str, ...] = arg_names
        self.required_args: frozenset[str] = required_args
        # Derive a human-readable ``argument_hint`` from the MCP-declared
        # argument schema when the prompt carries one. Required args are
        # wrapped in ``<angle>`` brackets (missing → error); optional args
        # get ``[square]`` brackets. ``None`` when the prompt takes no
        # arguments — mirrors SkillCommand's default and matches
        # claude-code's slash-command UX.
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
        """Parse the invocation tail into an ``arguments`` dict.

        Returns ``(arguments_dict, None)`` on success or
        ``(None, error_text)`` when required args are missing. Extra
        positionals beyond ``arg_names`` are dropped silently — lodash
        ``zipObject`` behaviour (claude-code parity). Tokens split on
        whitespace like ``str.split()`` (claude-code uses ``args.split(' ')``
        but whitespace split is strictly more forgiving for tab / multi-
        space input and doesn't change correct-case behaviour).
        """
        tokens = arg.split() if arg else []
        # Pair each declared argument name with its positional token. Names
        # beyond the provided tokens are considered "unprovided"; required
        # ones then flag the missing-arg error.
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
            # Required-arg validation failure: surface to the user and
            # journal it — no network call happens. Matches claude-code's
            # "surface an actionable error" shape for CLI prompt
            # commands. ``error`` is guaranteed non-None whenever
            # ``arguments`` is None (see ``_build_arguments`` contract).
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
            # The ``arguments`` kwarg exists on
            # ``MultiServerMCPClient.get_prompt`` (see the library's
            # ``client.py``: signature is
            # ``get_prompt(server_name, prompt_name, *, arguments=None)``).
            # Passing an empty dict is equivalent to ``None`` for
            # zero-arg prompts, so we always forward it for a single
            # code path.
            messages = await self._client.get_prompt(
                self._server,
                self._prompt,
                arguments=arguments,
            )
        except Exception as exc:  # noqa: BLE001
            # Server may have died between startup discovery and now. Surface
            # a user-visible failure + journal the reason rather than throw.
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
) -> _MCPPromptCommand:
    """Build a :class:`Command` that fetches and renders an MCP prompt.

    ``prompt_arguments`` is the raw list from ``mcp.types.Prompt.arguments``
    — each element is expected to expose ``.name: str`` and
    ``.required: bool | None`` (the MCP pydantic ``PromptArgument`` shape).
    When supplied, positional tokens passed to the slash command at
    invocation are forwarded to ``client.get_prompt(..., arguments=...)``.
    Defaults to ``None`` for backward compatibility: existing call sites
    that haven't wired the arg schema through continue to work — the prompt
    will still fetch, it just won't carry user-provided arguments.
    """
    arg_names: tuple[str, ...] = ()
    required_args: frozenset[str] = frozenset()
    if prompt_arguments:
        names: list[str] = []
        required: set[str] = set()
        for pa in prompt_arguments:
            name = getattr(pa, "name", None)
            if not isinstance(name, str) or not name:
                # Defensive: MCP spec requires a name but servers are
                # free to misbehave. Skipping nameless entries keeps the
                # positional index well-defined.
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
    )


def normalize_resource_contents(contents: Any) -> dict[str, Any]:
    """Flatten an MCP ``ResourceContents`` object into a JSON-safe dict.

    MCP defines two concrete subclasses:

    - ``TextResourceContents`` — has ``.text`` (str). We return it
      verbatim as ``{"type": "text", "text": ..., "mime": ..., "uri": ...}``
      so the LLM can read it directly.
    - ``BlobResourceContents`` — has ``.blob`` (base64 str). We return
      ``{"type": "blob", "mime": ..., "uri": ..., "size": <decoded
      length>}``. Deliberately DOES NOT echo the base64 payload back to
      the LLM: binary blobs are almost never useful for a text model and
      can be multi-MB. A future release can add an opt-in flag for tools
      that want the raw bytes (e.g. image understanding).

    Shape for unknown content classes degrades to a ``{"type": "unknown",
    "uri": ..., "mime": ..., "repr": ...}`` entry — still JSON-safe,
    still lets the LLM see that *something* came back.
    """
    import base64

    uri = getattr(contents, "uri", None)
    mime = getattr(contents, "mimeType", None)
    # Prefer text when available — it's the common case for MCP docs,
    # config files, and DB snapshots.
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
        # Compute the decoded byte size for context — helps the LLM decide
        # whether to ask for a different representation. Base64 decoding
        # failures fall back to the base64 string's length so we still
        # surface *something*.
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
