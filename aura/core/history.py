"""History helpers for the Aura agent loop.

Provides:
- openai_schema_for: converts an AuraTool to the OpenAI function-calling schema dict.
- serialize_messages: dumps a list of LangChain BaseMessages to plain dicts.
- deserialize_messages: rebuilds typed BaseMessage instances from those dicts.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from aura.tools.base import AuraTool

_MSG_TYPES: dict[str, type[BaseMessage]] = {
    "human": HumanMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
    "system": SystemMessage,
}


def openai_schema_for(tool: AuraTool) -> dict[str, Any]:
    """Return the OpenAI function-calling schema for *tool*."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_model.model_json_schema(),
        },
    }


def serialize_messages(msgs: list[BaseMessage]) -> list[dict[str, Any]]:
    """Dump *msgs* to a list of plain dicts suitable for JSON storage."""
    return [m.model_dump() for m in msgs]


def deserialize_messages(dicts: list[dict[str, Any]]) -> list[BaseMessage]:
    """Reconstruct typed BaseMessage instances from serialized *dicts*.

    Raises ValueError for any dict whose ``type`` field is missing or unknown.
    """
    out: list[BaseMessage] = []
    for d in dicts:
        t = d.get("type")
        cls = _MSG_TYPES.get(t) if isinstance(t, str) else None
        if cls is None:
            raise ValueError(f"unknown message type: {t!r}")
        out.append(cls.model_validate(d))
    return out
