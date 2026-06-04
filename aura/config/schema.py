"""AuraConfig pydantic v2 schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aura.domain.errors import AuraError


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
            "ask_user_question",
            "bash",
            "bash_background",
            "edit_file",
            "enter_plan_mode",
            "exit_plan_mode",
            "glob",
            "grep",
            "read_file",
            "skill",
            "task_create",
            "task_get",
            "task_list",
            "task_stop",
            "todo_write",
            "web_fetch",
            "write_file",
        ],
    )
    cleanup_completed_subagent_transcripts: bool = Field(
        default=False,
        description=(
            "Delete subagent transcript JSONL + meta.json after status=completed. "
            "Failed/cancelled transcripts are always kept for post-mortem."
        ),
    )
    mcp_overrides_builtin: bool = Field(
        default=False,
        description=(
            "Collision policy for builtin-vs-MCP tool name clashes. False: "
            "builtin wins, MCP dropped. True: MCP wins, builtin shadowed. "
            "Either way, mcp_tool_shadowed journal events log the winner."
        ),
    )


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default="~/.aura/sessions.db",
        description="SQLite path. May contain ~; expand via resolved_storage_path().",
    )


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: bool = Field(
        default=True,
        description="Render assistant text through rich.markdown.Markdown.",
    )
    buddy_enabled: bool = Field(
        default=True,
        description="Show pet buddy in the status bar (also honors AURA_NO_BUDDY=1).",
    )


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str = "~/.aura/logs/events.jsonl"


class WebSearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["duckduckgo"] = Field(
        default="duckduckgo",
        description="Search backend. Only duckduckgo (zero-config) ships today.",
    )
    api_key_env: str | None = Field(
        default=None,
        description="Env var holding the API key (ignored for duckduckgo).",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default cap when web_search omits max_results.",
    )


class RetryConfig(BaseModel):
    """Retry policy wrapping model.ainvoke() in the agent loop, not tool calls."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    base_delay_s: float = Field(default=1.0, gt=0)
    max_delay_s: float = Field(default=30.0, gt=0)


class CompactConfig(BaseModel):
    """Compaction tunables. Invariant: microcompact_keep_recent < trigger_pairs."""

    model_config = ConfigDict(extra="forbid")

    auto_threshold_buffer_tokens: int = Field(
        default=13_000,
        ge=0,
        description="Subtracted from context window to set the auto-compact threshold.",
    )
    max_files_to_restore: int = Field(
        default=5,
        ge=0,
        description="Cap on <recent-file> re-injects after a summary replaces history.",
    )
    max_tokens_per_file: int = Field(default=5_000, ge=0)
    max_summary_message_chars: int = Field(
        default=6_000,
        ge=0,
        description="Per-message cap while serialising history into the summary prompt.",
    )
    max_summary_tool_args_chars: int = Field(default=2_000, ge=0)
    fallback_summary_char_limit: int = Field(
        default=12_000,
        ge=0,
        description="Cap on the deterministic excerpt when no message fits the provider.",
    )
    max_summary_split_depth: int = Field(default=12, ge=1)
    max_consecutive_failures: int = Field(
        default=3,
        ge=1,
        description="Circuit breaker: N failed auto-compacts disable further auto firings.",
    )
    microcompact_trigger_pairs: int = Field(default=5, ge=0)
    microcompact_keep_recent: int = Field(default=3, ge=0)
    time_based_gap_threshold_minutes: int | None = Field(
        default=None,
        ge=1,
        description="If set, microcompact also fires after N min of assistant idle.",
    )


class TeamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Enable teams (multi-agent swarm). False: /team commands "
            "unregistered, send_message tool absent, AgentSession.join_team raises."
        ),
    )


class MCPServerConfig(BaseModel):
    """One MCP server; name namespaces its tools as mcp__<name>__<tool>."""

    model_config = ConfigDict(extra="forbid")

    name: str
    transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    f"MCP server {self.name!r}: 'command' is required for transport 'stdio'"
                )
            if self.url is not None:
                raise ValueError(
                    f"MCP server {self.name!r}: 'url' is not valid for "
                    "transport 'stdio'; remove it or switch transport"
                )
        else:  # sse, streamable_http
            if not self.url:
                raise ValueError(
                    f"MCP server {self.name!r}: 'url' is required for transport {self.transport!r}"
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
    teams: TeamsConfig = Field(default_factory=TeamsConfig)
    compact: CompactConfig = Field(default_factory=CompactConfig)
    retry: RetryConfig | None = Field(
        default=None,
        description="Retry for transient LLM errors; None = library defaults.",
    )
    context_window: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Override context window for the status-bar pressure ratio only "
            "(does not change what the model accepts)."
        ),
    )
    # Permission config lives in .aura/settings{,.local}.json so each file has one purpose.

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


class StatusLineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = ""
    timeout_ms: int = 500
    enabled: bool = True

    @field_validator("timeout_ms")
    @classmethod
    def _clamp_timeout(cls, v: int) -> int:
        if v < 50:
            return 50
        if v > 5000:
            return 5000
        return v

    @property
    def is_active(self) -> bool:
        return self.enabled and bool(self.command.strip())


class PermissionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["default", "bypass", "plan", "accept_edits"] = "default"
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    safety_exempt: list[str] = Field(default_factory=list)
    statusline: StatusLineConfig | None = None
    prompt_timeout_sec: float | None = Field(
        default=300.0,
        description=(
            "Seconds to wait for prompt response before treating as denial. "
            "None = wait forever; default 300 (5 minutes)."
        ),
    )
    disable_bypass: bool = Field(
        default=False,
        description="When true, refuse all attempts to enter bypass mode.",
    )
