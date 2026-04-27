#!/usr/bin/env bash
# One-shot launcher for the Aura desktop app.
#
# Usage:
#   ./desktop/run.sh           — interactive: prompts before installing missing deps
#   ./desktop/run.sh --yes     — non-interactive: auto-install everything missing
#   ./desktop/run.sh --build   — production build instead of dev shell

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSUME_YES=0
BUILD_MODE=0

for arg in "$@"; do
  case "$arg" in
    --yes|-y)   ASSUME_YES=1 ;;
    --build)    BUILD_MODE=1 ;;
    --help|-h)
      sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ── prompt helper ─────────────────────────────────────────────────────────────
ask() {
  local q="$1"
  if [[ $ASSUME_YES -eq 1 ]]; then return 0; fi
  read -r -p "$q [Y/n] " ans
  [[ -z "$ans" || "$ans" =~ ^[Yy]$ ]]
}

step() { printf "\033[1;36m▸\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m✓\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*" >&2; }

# ── 1. Rust toolchain ─────────────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
  warn "Rust (cargo) not found."
  if ask "Install Rust via rustup now? (~150 MB, takes 1-2 min)"; then
    step "Installing Rust toolchain via rustup…"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  else
    echo "Rust is required. Install manually: https://rustup.rs" >&2
    exit 1
  fi
else
  ok "Rust toolchain present ($(cargo --version))"
fi

# ── 2. tauri-cli ──────────────────────────────────────────────────────────────
if ! cargo tauri --version >/dev/null 2>&1; then
  warn "cargo-tauri not found."
  if ask "Install tauri-cli now? (~50 MB compile, takes 2-5 min)"; then
    step "Installing tauri-cli…"
    cargo install tauri-cli --version "^2.0" --locked
  else
    echo "tauri-cli is required. Install: cargo install tauri-cli --version '^2.0' --locked" >&2
    exit 1
  fi
else
  ok "tauri-cli present ($(cargo tauri --version 2>/dev/null | head -1))"
fi

# ── 3. uv (Python package manager) ────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found (needed for the Python headless subprocess)."
  if ask "Install uv now? (~30 MB)"; then
    step "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installer prints PATH instructions; pull it in for this session
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  else
    echo "uv is required. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
else
  ok "uv present ($(uv --version))"
fi

# ── 4. Python deps ────────────────────────────────────────────────────────────
# --extra all pulls in langchain-openai / -anthropic / -ollama / ddgs. Without
# them the headless subprocess crashes at import ("ModuleNotFoundError: No
# module named 'langchain_openai'") and the desktop shows "aura exited".
step "Syncing Python deps (uv sync --extra all)…"
(cd "$REPO_ROOT" && uv sync --extra all --quiet)
ok "Python deps ready"

# ── 5. Frontend deps ──────────────────────────────────────────────────────────
step "Installing frontend deps (npm install)…"
(cd "$SCRIPT_DIR/frontend" && npm install --silent --no-audit --no-fund)
ok "Frontend deps ready"

# ── 6. Port 5173 conflict — Vite's hardcoded port (matches tauri.conf.json) ───
#
# A prior `cargo tauri dev` that was killed via SIGINT/SIGTERM occasionally
# leaves an orphaned vite/node process holding port 5173. Detect + clear so
# the user doesn't have to deal with it manually.
if command -v lsof >/dev/null 2>&1; then
  PORT_PIDS="$(lsof -ti:5173 2>/dev/null || true)"
  if [[ -n "$PORT_PIDS" ]]; then
    warn "Port 5173 is in use by PID(s): $PORT_PIDS"
    # Show what's holding it so the user can sanity-check before nuking
    lsof -i:5173 -P -n 2>/dev/null | tail -n +2 | awk '{printf "    %s %s %s\n", $1, $2, $9}' || true
    if ask "Kill these processes so the dev server can bind?"; then
      step "Releasing port 5173…"
      # shellcheck disable=SC2086
      kill -9 $PORT_PIDS 2>/dev/null || true
      sleep 0.5  # give the OS a beat to release the socket
      ok "Port 5173 freed"
    else
      echo "Cannot start dev server while port 5173 is held. Aborting." >&2
      exit 1
    fi
  fi
fi

# ── 7. Launch ─────────────────────────────────────────────────────────────────
echo
if [[ $BUILD_MODE -eq 1 ]]; then
  step "Building production bundle (cargo tauri build)…"
  cd "$SCRIPT_DIR" && exec cargo tauri build
else
  step "Launching Aura desktop (cargo tauri dev)…"
  echo "  • First run compiles Rust deps (3-5 min). Subsequent runs are ~1s incremental."
  echo "  • Window will open automatically. Close it or Ctrl+C to stop."
  echo
  cd "$SCRIPT_DIR" && exec cargo tauri dev
fi
