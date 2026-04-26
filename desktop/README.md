# Aura Desktop

Tauri 2 + Rust desktop frontend for Aura. Sibling to `aura/` (the Python CLI), **not** under it — Tauri ships Rust + TS, not Python.

## Phase 1 — what's shipped

- **Rust IPC bridge** (`src-tauri/src/lib.rs`): on app start, spawns `python -m aura.cli.headless` (preferring `uv run` when available), pipes its stdout NDJSON event stream to Tauri's `aura-event` channel, and exposes `send_prompt(text)` / `stop_aura()` Tauri commands.
- **Headless Aura entry** (`aura/cli/headless.py`): single-tenant stdio mode that reads `{"kind":"prompt","text":"..."}` line-delimited requests from stdin and emits one event per line on stdout (assistant_delta / tool_call_started / final / error / etc.).
- **Vanilla TS frontend** (`frontend/src/main.ts`): minimal conversation UI — user bubble + streaming assistant bubble + tool-call cards + status indicator. ~200 LoC. No framework yet.

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
    │   ├── main.ts   Event subscriber + invoke caller
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

The dev shell starts Vite (frontend hot-reload) and a debug Tauri window. The Rust backend spawns `uv run python -m aura.cli.headless` from the repo root; ensure `uv sync` has been run there.

## Build (release)

```sh
cd desktop/frontend && npm run build && cd ..
cargo tauri build
```

The bundle target depends on platform: `.app` on macOS, `.msi`/`.exe` on Windows, `.AppImage`/`.deb` on Linux.

## Bridge contract — NDJSON over stdio

Each line on the headless subprocess's stdout is one JSON object:

| event | fields | meaning |
|---|---|---|
| `ready` | `session_id`, `model` | Emitted once at startup; bridge reports model name to status bar |
| `assistant_delta` | `text` | Streaming model output — append to active assistant bubble |
| `tool_call_started` | `name`, `args`, `id` | Tool dispatch begins; render a tool card |
| `tool_call_progress` | `id`, `chunk` | Mid-tool streaming chunk (currently dropped in Phase 1 UI) |
| `tool_call_completed` | `id`, `ok`, `result` | Tool returned; Phase 2 will render the result inline |
| `final` | `message`, `reason` | Turn ended (`reason` = `natural` / `aborted` / `max_turns`) |
| `error` | `message` | Fatal turn error (e.g. provider 5xx after retries exhausted) |
| `stderr` | `line` | Raw stderr from the subprocess (logged to console only) |
| `disconnected` | — | Emitted by the bridge when the subprocess exits |

User prompts go the other way as `{"kind":"prompt","text":"..."}` written to stdin via the `send_prompt` Tauri command.

## Phase 2 backlog

- Real frontend framework (React 19 or Solid 2)
- Status bar with token gauge + buddy + model selector
- Permission-prompt UI (replace stdin-based asker for desktop sessions)
- Slash-command palette + completion
- Tool result rendering (diff for `edit_file`, table for `task_list`, etc.)
- Multi-session tabs
- Settings panel (provider config, theme, keybindings)
- Auto-update via Tauri updater plugin
