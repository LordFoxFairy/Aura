"""Server-Sent Events framing helpers."""

from __future__ import annotations

import json
from typing import Any


def encode_sse(
    data: str,
    *,
    event: str | None = None,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """Encode a single SSE frame."""
    lines: list[str] = []
    if id is not None:
        lines.append(f"id: {id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def encode_json_sse(
    payload: dict[str, Any],
    *,
    event: str | None = None,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """Encode a compact JSON payload as an SSE frame."""
    data = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return encode_sse(data, event=event, id=id, retry=retry)
