"""AuraConfig pydantic v2 schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aura.errors import AuraError


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    protocol: Literal["openai", "anthropic", "ollama"]
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    models: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(
        default_factory=lambda: [
            "bash", "bash_background", "edit_file",
            "enter_plan_mode", "exit_plan_mode",
            "glob", "grep", "read_file",
            "task_get", "task_list", "task_stop",
            "todo_write", "web_fetch", "write_file",
        ],
    )


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default="~/.aura/sessions.db",
        description="Path to the SQLite DB. May contain ~; expand via resolved_storage_path().",
    )


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "default"


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str = "~/.aura/logs/events.jsonl"


class WebSearchConfig(BaseModel):
    """Backend selection for the ``web_search`` tool.

    ``provider`` picks the backend; ``duckduckgo`` is the zero-config default
    and the only backend this release supports.

    ``api_key_env`` names the env var that holds the key (the key itself is
    never stored in config). Ignored for the DuckDuckGo backend.

    ``max_results`` is the default cap when the caller invokes ``web_search``
    without an explicit ``max_results`` argument (i.e. the schema default
    fires). An explicit argument always wins.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["duckduckgo"] = "duckduckgo"
    api_key_env: str | None = None
    max_results: int = Field(default=5, ge=1, le=20)


class MCPServerConfig(BaseModel):
    """One MCP server entry. The ``name`` namespaces tools as
    ``mcp__<name>__<tool>`` and commands as ``/<name>__<prompt>``.

    Three transports are supported, matching ``langchain-mcp-adapters``:

    - ``stdio`` (default): spawn a child process; requires ``command``.
    - ``sse``: Server-Sent Events HTTP endpoint; requires ``url``.
    - ``streamable_http``: streamable HTTP endpoint (the successor to SSE in
      the upstream spec); requires ``url``.

    For network transports, ``headers`` is passed through verbatim — this is
    where bearer tokens / API keys go. For ``stdio`` the ``env`` dict is used
    instead and ``url``/``headers`` are ignored.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    # Populated for stdio transport; unused (and must be None) for http/sse.
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    # Populated for sse / streamable_http; must be None for stdio.
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"MCP server {self.name!r}: 'command' is required for "
                    "transport 'stdio'"
                )
            if self.url is not None:
                raise ValueError(
                    f"MCP server {self.name!r}: 'url' is not valid for "
                    "transport 'stdio'; remove it or switch transport"
                )
        else:  # sse, streamable_http
            if not self.url:
                raise ValueError(
                    f"MCP server {self.name!r}: 'url' is required for "
                    f"transport {self.transport!r}"
                )
            if self.command is not None:
                raise ValueError(
                    f"MCP server {self.name!r}: 'command' is not valid for "
                    f"transport {self.transport!r}; use 'url' instead"
                )
        return self


class AuraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [
            ProviderConfig(name="openai", protocol="openai", api_key_env="OPENAI_API_KEY"),
        ],
    )
    router: dict[str, str] = Field(default_factory=lambda: {"default": "openai:gpt-4o-mini"})
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    web_search: WebSearchConfig | None = None
    # Optional per-user override for the context window the status bar uses
    # to render the live context-pressure ratio. When ``None``, Aura looks
    # the window up by model spec via ``aura.core.llm.get_context_window``;
    # when set, this value wins regardless of model. Useful for:
    #  - frontier models not yet in the table that the user knows the exact
    #    window size of
    #  - beta / extended-context deployments (e.g. Claude 4.x with 1M
    #    extended context enabled — the model spec is the same but the
    #    window is 5×)
    #  - proxies that round-trip through a different provider's tokenizer
    # Does NOT change what the model actually accepts — only the denominator
    # the status bar divides by.
    context_window: int | None = Field(default=None, gt=0)
    # NOTE: permission config does NOT live here. Providers/router/storage/log
    # are runtime wiring; permissions are a separate concern with their own
    # file(s) at ``.aura/settings.json`` + ``.aura/settings.local.json``,
    # loaded by ``aura.core.permissions.store.load``. Keeping them separate
    # means each file has ONE purpose and the user knows exactly which file
    # to edit. See spec §7.

    @model_validator(mode="after")
    def _validate_cross_refs(self) -> AuraConfig:
        names = [p.name for p in self.providers]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate provider names: {sorted(dupes)}")

        if "default" not in self.router:
            raise ValueError("router must contain a 'default' entry")

        known = set(names)
        for alias, target in self.router.items():
            if ":" not in target:
                raise ValueError(f"router[{alias!r}]={target!r} must be 'provider:model'")
            provider_name = target.split(":", 1)[0]
            if provider_name not in known:
                raise ValueError(
                    f"router[{alias!r}]={target!r} references unknown provider {provider_name!r}; "
                    f"known: {sorted(known)}"
                )
        return self

    def resolved_storage_path(self) -> Path:
        return Path(self.storage.path).expanduser()


class AuraConfigError(AuraError):
    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail
