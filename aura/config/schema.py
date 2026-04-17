"""AuraConfig pydantic v2 schema."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str = "openai"
    name: str = "gpt-4o-mini"


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=lambda: ["read_file", "write_file", "bash"])


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "~/.aura/sessions.db"


class UIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    theme: str = "default"


class AuraConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    def resolved_storage_path(self) -> Path:
        return Path(self.storage.path).expanduser()


class AuraConfigError(Exception):
    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source
        self.detail = detail
