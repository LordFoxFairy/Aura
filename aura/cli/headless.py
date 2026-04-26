"""Headless NDJSON entry point for desktop / external-frontend integrations.

Reads line-delimited JSON requests from stdin, emits one JSON event per line
to stdout. Each request is a ``{"kind": "prompt", "text": "..."}`` envelope;
each response is the JSON-serialized event from :mod:`aura.schemas.events`
plus a small ``{"event": "<name>"}`` discriminator.

Designed for the Tauri desktop frontend that spawns ``python -m
aura.cli.headless`` as a subprocess and pipes user prompts down stdin while
streaming agent events back up stdout. Pure stdio — no prompt_toolkit,
no rich, no terminal control sequences. Stderr is reserved for fatal
errors / startup banners; routine errors come through stdout as
``{"event": "error", "message": ...}`` records.

Event shapes (all NDJSON, one per line):

- ``{"event": "ready", "session_id": "..."}`` — emitted once at startup
- ``{"event": "assistant_delta", "text": "..."}`` — streaming model text
- ``{"event": "tool_call_started", "name": "...", "args": {...}, "id": "..."}``
- ``{"event": "tool_call_progress", "id": "...", "chunk": {...}}``
- ``{"event": "tool_call_completed", "id": "...", "ok": bool, "result": {...}}``
- ``{"event": "final", "message": "...", "reason": "..."}`` — turn ended
- ``{"event": "error", "message": "..."}`` — fatal turn error
- ``{"event": "exited"}`` — emitted right before the process closes stdin

The CLI is single-tenant (one Agent per process, one prompt at a time).
For multi-session use, the desktop spawns multiple subprocesses.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aura.config.loader import load_config
from aura.core.agent import Agent
from aura.core.llm import make_model_for_spec
from aura.core.persistence.storage import SessionStorage
from aura.schemas.events import (
    AssistantDelta,
    Final,
    ToolCallCompleted,
    ToolCallProgress,
    ToolCallStarted,
)


def _emit(payload: dict[str, Any]) -> None:
    """Write one NDJSON line to stdout + flush.

    Tauri reads stdout line-by-line; an unflushed line stays in Python's
    pipe buffer, leaving the frontend "stuck" mid-turn. Flush after every
    write so streaming feels real-time.
    """
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Map an :mod:`aura.schemas.events` dataclass to its NDJSON shape.

    Field names mirror the dataclass attributes verbatim so the frontend
    type definitions stay aligned with the Python source of truth.
    """
    if isinstance(event, AssistantDelta):
        return {"event": "assistant_delta", "text": event.text}
    if isinstance(event, ToolCallStarted):
        return {
            "event": "tool_call_started",
            "name": event.name,
            "input": event.input,
        }
    if isinstance(event, ToolCallProgress):
        return {
            "event": "tool_call_progress",
            "name": event.name,
            "stream": event.stream,
            "chunk": event.chunk,
        }
    if isinstance(event, ToolCallCompleted):
        return {
            "event": "tool_call_completed",
            "name": event.name,
            "output": event.output,
            "error": event.error,
        }
    if isinstance(event, Final):
        return {
            "event": "final",
            "message": event.message,
            "reason": getattr(event, "reason", "natural"),
        }
    # Unknown event types — surface the class name so the frontend can
    # fall back to a generic "info" line instead of swallowing.
    return {"event": "unknown", "type": type(event).__name__}


async def _run() -> int:
    cfg = load_config()
    spec = cfg.router.get("default", "")
    if not spec:
        _emit({
            "event": "error",
            "message": "config.router['default'] is missing — cannot start headless",
        })
        return 1
    model = make_model_for_spec(spec, cfg)
    storage_path = Path(cfg.storage.path).expanduser()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = SessionStorage(storage_path)
    agent = Agent(config=cfg, model=model, storage=storage)
    _emit({"event": "ready", "session_id": agent.session_id, "model": spec})

    try:
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            line = await reader.readline()
            if not line:
                # stdin closed — the frontend exited.
                break
            try:
                request = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError as exc:
                _emit({"event": "error", "message": f"bad request: {exc}"})
                continue
            if request.get("kind") != "prompt":
                _emit({
                    "event": "error",
                    "message": f"unsupported request kind: {request.get('kind')!r}",
                })
                continue
            text = request.get("text", "")
            if not isinstance(text, str) or not text:
                _emit({"event": "error", "message": "empty prompt"})
                continue
            try:
                async for event in agent.astream(text):
                    _emit(_event_to_dict(event))
            except Exception as exc:  # noqa: BLE001
                _emit({
                    "event": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                })
    finally:
        await agent.aclose()
        _emit({"event": "exited"})
    return 0


def main() -> int:
    """Entry point — bootstrap an Agent and pipe stdin → astream → stdout."""
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        _emit({"event": "exited"})
        return 130


if __name__ == "__main__":
    sys.exit(main())
