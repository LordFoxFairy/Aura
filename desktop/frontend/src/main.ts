/**
 * Aura desktop frontend — Phase 1.
 *
 * Subscribes to the Rust-side ``aura-event`` event stream (one event per
 * NDJSON line emitted by the headless Aura subprocess) and renders a
 * minimal conversation pane. Sends user prompts via the ``send_prompt``
 * Tauri command.
 *
 * Phase 2 will add: a real framework (React 19), per-tool-call cards
 * with diff rendering, permission-prompt UI, slash-command palette,
 * status bar with token gauge.
 */

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./style.css";

interface AuraEvent {
  event: string;
  text?: string;
  message?: string;
  reason?: string;
  name?: string;
  input?: unknown;
  output?: unknown;
  error?: string | null;
  stream?: string;
  chunk?: string;
  session_id?: string;
  model?: string;
  line?: string;
  type?: string;
  // permission_request fields
  id?: string;
  tool?: string;
  args?: unknown;
  rule_hint?: string;
  is_destructive?: boolean;
}

const conversationEl = document.getElementById("conversation") as HTMLElement;
const statusEl = document.getElementById("status") as HTMLElement;
const inputEl = document.getElementById("input") as HTMLTextAreaElement;
const sendBtn = document.getElementById("send") as HTMLButtonElement;

const permOverlay = document.getElementById("permission-overlay") as HTMLElement;
const permModal = permOverlay.querySelector(".permission-modal") as HTMLElement;
const permToolEl = document.getElementById("permission-tool") as HTMLElement;
const permArgsEl = document.getElementById("permission-args") as HTMLElement;
const permHintEl = document.getElementById("permission-hint") as HTMLElement;
const permDenyBtn = document.getElementById("perm-deny") as HTMLButtonElement;
const permOnceBtn = document.getElementById("perm-once") as HTMLButtonElement;
const permAlwaysBtn = document.getElementById("perm-always") as HTMLButtonElement;

/** ID of the permission request currently displayed in the modal, or null. */
let activePermissionId: string | null = null;

/** Track the assistant message currently being streamed so deltas append in place. */
let activeAssistantBubble: HTMLElement | null = null;

function appendUserBubble(text: string): void {
  const bubble = document.createElement("div");
  bubble.className = "bubble user";
  bubble.textContent = text;
  conversationEl.appendChild(bubble);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function ensureAssistantBubble(): HTMLElement {
  if (activeAssistantBubble === null) {
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    conversationEl.appendChild(bubble);
    activeAssistantBubble = bubble;
  }
  return activeAssistantBubble;
}

function appendAssistantDelta(text: string): void {
  const bubble = ensureAssistantBubble();
  bubble.textContent = (bubble.textContent ?? "") + text;
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function finalizeAssistantBubble(reason: string): void {
  if (activeAssistantBubble !== null && reason !== "natural") {
    const tag = document.createElement("span");
    tag.className = "reason-tag";
    tag.textContent = ` [${reason}]`;
    activeAssistantBubble.appendChild(tag);
  }
  activeAssistantBubble = null;
}

function appendToolCallCard(name: string, args: unknown): void {
  const card = document.createElement("div");
  card.className = "tool-card";
  const head = document.createElement("div");
  head.className = "tool-head";
  head.textContent = `🔧 ${name}`;
  const body = document.createElement("pre");
  body.className = "tool-args";
  body.textContent = JSON.stringify(args, null, 2);
  card.appendChild(head);
  card.appendChild(body);
  conversationEl.appendChild(card);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function appendError(message: string): void {
  const bubble = document.createElement("div");
  bubble.className = "bubble error";
  bubble.textContent = `error: ${message}`;
  conversationEl.appendChild(bubble);
  conversationEl.scrollTop = conversationEl.scrollHeight;
}

function setStatus(text: string, kind: "ready" | "thinking" | "error" | "off"): void {
  statusEl.textContent = text;
  statusEl.dataset.kind = kind;
}

function showPermissionPrompt(ev: AuraEvent): void {
  if (!ev.id || !ev.tool) return;
  activePermissionId = ev.id;
  permToolEl.textContent = ev.tool;
  permArgsEl.textContent = JSON.stringify(ev.input ?? ev.args ?? {}, null, 2);
  permHintEl.textContent = ev.rule_hint
    ? `"Always" installs rule: ${ev.rule_hint}`
    : "";
  permModal.dataset.destructive = String(ev.is_destructive !== false);
  permOverlay.hidden = false;
}

function hidePermissionPrompt(): void {
  activePermissionId = null;
  permOverlay.hidden = true;
}

async function respondPermission(
  choice: "accept" | "always" | "deny",
): Promise<void> {
  if (activePermissionId === null) return;
  const id = activePermissionId;
  hidePermissionPrompt();
  try {
    await invoke("send_permission_response", { id, choice, feedback: "" });
  } catch (e) {
    appendError(`permission response failed: ${e}`);
  }
}

permDenyBtn.addEventListener("click", () => void respondPermission("deny"));
permOnceBtn.addEventListener("click", () => void respondPermission("accept"));
permAlwaysBtn.addEventListener("click", () => void respondPermission("always"));

async function send(): Promise<void> {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  appendUserBubble(text);
  setStatus("thinking…", "thinking");
  try {
    await invoke("send_prompt", { text });
  } catch (e) {
    appendError(String(e));
    setStatus("ready", "ready");
  }
}

sendBtn.addEventListener("click", () => {
  void send();
});

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    void send();
  }
});

/** Wire the ``aura-event`` stream. Each NDJSON event becomes a UI update. */
void listen<AuraEvent>("aura-event", (msg) => {
  const ev = msg.payload;
  switch (ev.event) {
    case "ready":
      setStatus(`ready · ${ev.model ?? "?"}`, "ready");
      break;
    case "assistant_delta":
      if (typeof ev.text === "string") appendAssistantDelta(ev.text);
      break;
    case "tool_call_started":
      finalizeAssistantBubble("natural");
      appendToolCallCard(ev.name ?? "(unknown)", ev.input ?? {});
      break;
    case "permission_request":
      showPermissionPrompt(ev);
      break;
    case "tool_call_completed":
      // Phase 1: just dim the most recent tool card on completion. Phase 2
      // will show structured success/failure + result preview.
      break;
    case "final":
      finalizeAssistantBubble(ev.reason ?? "natural");
      setStatus("ready", "ready");
      break;
    case "error":
      appendError(ev.message ?? "(no message)");
      setStatus("error", "error");
      break;
    case "stderr":
      // Don't render stderr inline — it's noisy. Phase 2 adds a debug pane.
      console.warn("[aura stderr]", ev.line);
      break;
    case "disconnected":
      setStatus("aura exited", "off");
      break;
    default:
      // Drop unknown events silently in Phase 1; log for debugging.
      console.debug("[aura event]", ev);
  }
});
