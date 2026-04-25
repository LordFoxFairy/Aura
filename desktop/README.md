# desktop/

Tauri-based desktop frontend for Aura. Sibling to `aura/` (the Python CLI), **not** under it — Tauri ships Rust + TS/JS, which is not a Python module.

## Layout

```
desktop/
  src-tauri/        # Rust backend (Tauri commands, IPC bridge to aura CLI)
    src/
    Cargo.toml      # TODO: cargo tauri init
    tauri.conf.json # TODO
  frontend/         # TS/JS UI (any framework — React / Solid / Svelte)
    src/
    package.json    # TODO
```

## Bridge to `aura` CLI

The desktop app spawns the `aura` Python process as a subprocess and communicates over stdio (NDJSON event stream — same shape `aura.schemas.events` emits in the REPL). No shared memory; the desktop is a thin presentation layer.

## Status

Scaffold only — Tauri project not yet initialized. To bootstrap:

```sh
cd desktop
cargo install tauri-cli --version "^2"
cargo tauri init       # generates Cargo.toml + tauri.conf.json + src-tauri/src/main.rs
cd frontend && npm init -y
```

Build artifacts (`target/`, `node_modules/`, `dist/`) belong in `.gitignore`.
