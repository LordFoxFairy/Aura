//! Aura desktop — Rust backend.
//!
//! Spawns ``python -m desktop.host.headless`` as a child process, streams
//! SSE-framed events from its stdout into Tauri events the frontend
//! subscribes to via ``listen("aura-event", ...)``. User prompts come
//! down via the ``send_prompt`` command which writes one NDJSON request
//! per call to the child's stdin (request channel stays NDJSON; only
//! the event stream is SSE).
//!
//! Single-tenant: one Aura subprocess per app instance. Restart the
//! app for a fresh agent. Multi-session UX is Phase 2.

use std::process::Stdio;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::Mutex;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PromptRequest {
    kind: String,
    text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct PermissionResponse {
    kind: String,
    id: String,
    choice: String,        // "accept" | "always" | "deny"
    feedback: String,      // optional free-text from the user
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<String>, // "project" | "session" — only meaningful with choice="always"
}

/// Shared handle on the running Aura subprocess. Wrapped in a Mutex so
/// the ``send_prompt`` command can lock the stdin half across awaits.
struct AuraProcess {
    /// stdin half — written one NDJSON request per ``send_prompt`` call.
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    /// keeps the child alive; killed in ``stop_aura`` or on drop.
    _child: Arc<Mutex<Option<Child>>>,
}

#[tauri::command]
async fn send_prompt(
    text: String,
    state: State<'_, AuraProcess>,
) -> Result<(), String> {
    let req = PromptRequest {
        kind: "prompt".to_string(),
        text,
    };
    let line =
        serde_json::to_string(&req).map_err(|e| format!("encode: {e}"))? + "\n";
    let mut guard = state.stdin.lock().await;
    let stdin = guard
        .as_mut()
        .ok_or_else(|| "aura subprocess not running".to_string())?;
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| format!("write stdin: {e}"))?;
    stdin
        .flush()
        .await
        .map_err(|e| format!("flush stdin: {e}"))?;
    Ok(())
}

#[tauri::command]
async fn send_permission_response(
    id: String,
    choice: String,
    feedback: Option<String>,
    scope: Option<String>,
    state: State<'_, AuraProcess>,
) -> Result<(), String> {
    // Reject unrecognised values here so a buggy frontend can't smuggle
    // them to the Python asker (which would silently degrade scope).
    if !matches!(choice.as_str(), "accept" | "always" | "deny") {
        return Err(format!("invalid choice: {:?}", choice));
    }
    if let Some(ref s) = scope {
        if !matches!(s.as_str(), "project" | "session") {
            return Err(format!("invalid scope: {:?}", s));
        }
    }
    let req = PermissionResponse {
        kind: "permission_response".to_string(),
        id,
        choice,
        feedback: feedback.unwrap_or_default(),
        scope,
    };
    let line =
        serde_json::to_string(&req).map_err(|e| format!("encode: {e}"))? + "\n";
    let mut guard = state.stdin.lock().await;
    let stdin = guard
        .as_mut()
        .ok_or_else(|| "aura subprocess not running".to_string())?;
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| format!("write stdin: {e}"))?;
    stdin
        .flush()
        .await
        .map_err(|e| format!("flush stdin: {e}"))?;
    Ok(())
}

#[tauri::command]
async fn stop_aura(state: State<'_, AuraProcess>) -> Result<(), String> {
    // Drop the stdin half — that's the graceful shutdown signal for the
    // headless loop (it sees stdin EOF and exits its readline loop).
    {
        let mut guard = state.stdin.lock().await;
        *guard = None;
    }
    // Force-kill if the child didn't exit on its own within 2s.
    let mut child_guard = state._child.lock().await;
    if let Some(mut child) = child_guard.take() {
        let kill_timeout = tokio::time::sleep(std::time::Duration::from_secs(2));
        tokio::select! {
            _ = child.wait() => {}
            _ = kill_timeout => {
                let _ = child.kill().await;
            }
        }
    }
    Ok(())
}

/// Spawn the aura headless subprocess and wire its stdout → tauri event stream.
async fn spawn_aura(app: AppHandle) -> Result<AuraProcess, String> {
    // Locate the repo root so we can run ``python -m desktop.host.headless``.
    // src-tauri/ → desktop/ → repo root.
    let repo_root = std::env::current_dir()
        .map_err(|e| format!("current_dir: {e}"))?
        .parent()
        .ok_or_else(|| "no parent dir".to_string())?
        .parent()
        .ok_or_else(|| "no grandparent dir".to_string())?
        .to_path_buf();

    // Prefer ``uv run`` so the project's pinned Python + dependencies
    // are picked up automatically. Fall back to bare ``python -m`` if
    // ``uv`` isn't on PATH (rare on dev machines but covered).
    let (program, args): (&str, Vec<&str>) = if which::which("uv").is_ok() {
        ("uv", vec!["run", "python", "-m", "desktop.host.headless"])
    } else {
        ("python", vec!["-m", "desktop.host.headless"])
    };

    let mut child = Command::new(program)
        .args(&args)
        .current_dir(&repo_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn aura: {e}"))?;

    let stdin = child.stdin.take().ok_or("child stdin missing")?;
    let stdout = child.stdout.take().ok_or("child stdout missing")?;
    let stderr = child.stderr.take().ok_or("child stderr missing")?;

    // Stream stdout: SSE frames ("event: aura\ndata: <json>\n\n") →
    // one Tauri "aura-event" per frame. Accumulate the data line(s) of
    // each frame until a blank-line delimiter, then strip the
    // ``data: `` prefix and parse the JSON payload. ``event:``,
    // ``id:``, ``retry:`` lines are ignored — we only use ``data:``.
    let app_clone = app.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        let mut data_buf = String::new();
        while let Ok(Some(line)) = reader.next_line().await {
            if line.is_empty() {
                if data_buf.is_empty() {
                    continue;
                }
                let payload: serde_json::Value = serde_json::from_str(&data_buf)
                    .unwrap_or_else(|_| {
                        serde_json::json!({"event": "raw", "line": data_buf.clone()})
                    });
                let _ = app_clone.emit("aura-event", payload);
                data_buf.clear();
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                data_buf.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            }
            // event:, id:, retry: and any other field lines are dropped.
        }
        let _ = app_clone.emit(
            "aura-event",
            serde_json::json!({"event": "disconnected"}),
        );
    });

    // Forward stderr lines as ``stderr`` events (debug surface).
    let app_err = app.clone();
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = reader.next_line().await {
            if line.is_empty() {
                continue;
            }
            let _ = app_err.emit(
                "aura-event",
                serde_json::json!({"event": "stderr", "line": line}),
            );
        }
    });

    Ok(AuraProcess {
        stdin: Arc::new(Mutex::new(Some(stdin))),
        _child: Arc::new(Mutex::new(Some(child))),
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Spawn the aura subprocess at startup. Failures surface
            // through the same event stream so the frontend shows a
            // banner instead of a silent dead app.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                match spawn_aura(handle.clone()).await {
                    Ok(proc) => {
                        handle.manage(proc);
                    }
                    Err(err) => {
                        let _ = handle.emit(
                            "aura-event",
                            serde_json::json!({
                                "event": "error",
                                "message": format!("failed to spawn aura: {err}"),
                            }),
                        );
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_prompt,
            send_permission_response,
            stop_aura,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
