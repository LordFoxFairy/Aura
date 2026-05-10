# Aura Desktop

Tauri 2 + Rust + React desktop frontend for Aura. Sibling to `aura/` (the Python CLI), **not** under it — Tauri ships Rust + TSX, not Python.

## Phase 1 — what's shipped

- **Rust IPC bridge** (`src-tauri/src/lib.rs`): on app start, spawns `python -m aura.desktop.headless` (preferring `uv run` when available), pipes its stdout NDJSON event stream to Tauri's `aura-event` channel, and exposes `send_prompt(text)`, `send_permission_response(...)`, and `stop_aura()` Tauri commands.
- **Headless Aura entry** (`aura/desktop/headless.py`): single-tenant stdio mode that reads prompt and permission-response requests from stdin and emits one event per line on stdout. It serializes internal `AgentEvent` values through `aura.transport.wire` so desktop, future HTTP/SSE surfaces, and AG-UI adapters share the same event contract.
- **Shared transport layer** (`aura/transport/`): `wire.py` defines Aura's stable JSON event shape, `agui.py` maps that shape to AG-UI-style lifecycle/text/tool/state events, `sse.py` frames JSON payloads for outbound Server-Sent Events, and `stream.py` adapts `Agent.astream(...)` into those transports.
- **React frontend** (`frontend/src/main.tsx`, `frontend/src/components/`): conversation UI with streaming assistant bubbles, tool-call cards, status surfaces, and desktop permission prompts.

## Layout

```
desktop/
├── src-tauri/        Rust backend (Tauri commands, IPC bridge, subprocess lifecycle)
│   ├── src/
│   │   ├── lib.rs    Bridge logic — spawn, stream, send, stop
│   │   └── main.rs   Entry stub
│   ├── Cargo.toml
│   └── tauri.conf.json
└── frontend/         TS UI (vite-bundled)
    ├── src/
    │   ├── main.tsx  React entry point
    │   ├── bridge.ts Tauri event subscriber + invoke callers
    │   ├── store.ts  Client-side conversation/status state
    │   ├── types.ts  Shared bridge and UI contracts
    │   ├── components/
    │   └── style.css
    ├── index.html
    ├── package.json
    └── vite.config.ts
```

## Run (dev)

```sh
# 1. Install JS deps once
cd desktop/frontend
npm install

# 2. Run the dev shell (auto-starts Vite + Tauri window).
cd ..               # back to desktop/
cargo tauri dev
```

The dev shell starts Vite (frontend hot-reload) and a debug Tauri window. The Rust backend spawns `uv run python -m aura.desktop.headless` from the repo root; ensure `uv sync` has been run there.

## Build (release)

```sh
cd desktop/frontend && npm run build && cd ..
cargo tauri build
```

The bundle target depends on platform: `.app` on macOS, `.msi`/`.exe` on Windows, `.AppImage`/`.deb` on Linux.

## Bridge contract

### Python stdout NDJSON

Each line on the headless subprocess's stdout is one JSON object produced by `aura.transport.wire`:

| event | fields | meaning |
|---|---|---|
| `ready` | `session_id`, `model` | Emitted once at startup; bridge reports model name to status bar |
| `assistant_delta` | `text` | Streaming model output — append to active assistant bubble |
| `tool_call_started` | `id?`, `name`, `input` | Tool dispatch begins; render a tool card |
| `tool_call_progress` | `id?`, `name`, `stream`, `chunk` | Mid-tool streaming chunk (`stream` = `stdout` / `stderr`) |
| `tool_call_completed` | `id?`, `name`, `content` | Tool returned; `content = {text, error}` carries one shape for both success and failure. `error: true` flags a failed dispatch. |
| `final` | `message`, `reason` | Turn ended (`reason` = `natural` / `aborted` / `max_turns`) |
| `permission_request` | `id`, `tool`, `args`, `rule_hint`, `is_destructive` | Headless permission prompt request; frontend responds with `permission_response` |
| `permission_audit` | `tool`, `text` | Permission decision audit entry emitted by the core loop |
| `aura_state` | `model`, `mode`, `cwd`, `tokens`, `pinned`, `window`, `last_turn_seconds` | Status snapshot emitted at startup and after every turn |
| `unknown` | `type` | Fallback for event types the headless bridge does not explicitly map |
| `exited` | — | Emitted by the headless process before shutdown |
| `error` | `message` | Fatal turn error (e.g. provider 5xx after retries exhausted) |

User prompts go the other way as `{"kind":"prompt","text":"..."}` written to stdin via the `send_prompt` Tauri command.
Permission responses go back as `{"kind":"permission_response","id":"...","choice":"accept|always|deny","feedback":"..."}`.
Tool events should be correlated by `id` whenever present. Legacy tool events may omit `id`; the frontend generates a local id for rendering and falls back to the most recent incomplete tool with the same name.

### AG-UI and SSE

External HTTP-style integrations should use the shared adapter chain instead of reading `AgentEvent` directly:

```text
Agent.astream(prompt)
  -> aura.transport.wire.event_to_wire(...)
  -> aura.transport.agui.AguiAdapter.convert(...)
  -> aura.transport.sse.encode_json_sse(...)
```

`aura.transport.sse` is outbound Server-Sent Event framing for Aura run events. It is separate from any inbound MCP server transport named SSE.

### Rust bridge events

The Rust bridge forwards parsed Python stdout records as-is on Tauri's `aura-event` channel and adds these local transport events:

| event | fields | meaning |
|---|---|---|
| `raw` | `line` | Python stdout line that was not valid JSON |
| `stderr` | `line` | Raw stderr from the subprocess (logged to console only) |
| `disconnected` | — | Emitted by the bridge when the subprocess exits |

## Phase 2 backlog

- Mode toggle + permission history polish
- Status bar with token gauge + model selector
- Slash-command palette + completion
- Tool result rendering (diff for `edit_file`, table for `task_list`, etc.)
- Multi-session tabs
- Settings panel (provider config, theme, keybindings)
- Auto-update via Tauri updater plugin
