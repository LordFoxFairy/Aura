"""Token estimation — CJK-aware fallback for providers that omit usage."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage


def estimate_text_tokens(text: str) -> int:
    ascii_chars = 0
    cjk_chars = 0
    other_non_ascii = 0
    for ch in text:
        code = ord(ch)
        if code < 128:
            ascii_chars += 1
        elif (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk_chars += 1
        else:
            other_non_ascii += 1
    return max(1, (ascii_chars + 3) // 4 + cjk_chars + (other_non_ascii + 1) // 2)


def estimate_json_tokens(value: object) -> int:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return estimate_text_tokens(text)


def estimate_message_tokens(
    message: BaseMessage,
    *,
    include_tool_calls: bool = True,
) -> int:
    content = message.content
    total = estimate_text_tokens(content if isinstance(content, str) else str(content))
    if include_tool_calls and isinstance(message, AIMessage):
        for tool_call in message.tool_calls or []:
            total += estimate_text_tokens(str(tool_call.get("name") or ""))
            total += estimate_json_tokens(tool_call.get("args") or {})
    return total
