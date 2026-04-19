"""Tool-schema bridge for binding tools to LangChain chat models.

Messages serialization lives in langchain_core.messages directly
(messages_to_dict / messages_from_dict) — no wrapper needed.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class _SchemaTool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]


def tool_schema_for(tool: _SchemaTool) -> dict[str, Any]:
    """LangChain bind_tools 接受这个 dict 形状；各 provider 绑定层内部翻译成 native tool 格式。

    我们手搓而非用 ``convert_to_openai_tool``：后者从 Pydantic 类取不到我们的
    ``tool.name`` / ``tool.description``（它们是项目元数据，不在 input_model 上）。
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_model.model_json_schema(),
        },
    }
