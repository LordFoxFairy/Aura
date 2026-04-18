"""AuraConfig pydantic v2 schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    protocol: Literal["openai", "anthropic", "ollama"]
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    models: list[str] = Field(default_factory=list)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=lambda: ["read_file", "write_file", "bash"])


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        default="~/.aura/sessions.db",
        description="Path to the SQLite DB. May contain ~; expand via resolved_storage_path().",
    )


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "default"


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

    @model_validator(mode="after")
    def _validate_cross_refs(self) -> AuraConfig:
        # 1) Unique provider names
        names = [p.name for p in self.providers]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate provider names: {sorted(dupes)}")

        # 2) router must have 'default'
        if "default" not in self.router:
            raise ValueError("router must contain a 'default' entry")

        # 3) Every router value must resolve: split on first ':', left must be a known provider name
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


class AuraConfigError(Exception):
    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail
