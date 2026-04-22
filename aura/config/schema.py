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
            "bash", "edit_file", "glob", "grep", "read_file",
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
    and the only one actually implemented in this release. ``tavily`` and
    ``serper`` slots exist so config validates (and users can pin a future
    backend without a schema migration) — invoking them raises a clear
    ``ToolError`` telling the user the backend is not yet implemented.

    ``api_key_env`` names the env var that holds the key (the key itself is
    never stored in config). Ignored for the DuckDuckGo backend.

    ``max_results`` is the default cap when the caller invokes ``web_search``
    without an explicit ``max_results`` argument (i.e. the schema default
    fires). An explicit argument always wins.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["duckduckgo", "tavily", "serper"] = "duckduckgo"
    api_key_env: str | None = None
    max_results: int = Field(default=5, ge=1, le=20)


class MCPServerConfig(BaseModel):
    """One MCP server entry. The ``name`` namespaces tools as
    ``mcp__<name>__<tool>`` and commands as ``/<name>__<prompt>``.

    We only support stdio for this release; ``transport`` is kept as a field
    so adding SSE/HTTP later is a non-breaking schema change.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    transport: Literal["stdio"] = "stdio"
    enabled: bool = True


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
