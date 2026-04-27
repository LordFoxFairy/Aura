/**
 * Shared type definitions for the Aura desktop frontend.
 *
 * These are pure data types — no React, no DOM, no side effects.
 * Every shape here mirrors the wire format emitted by aura/desktop/headless.py.
 */

/** Discriminated union of every event kind delivered over the ``aura-event`` Tauri stream. */
export interface AuraReadyEvent        { event: "ready"; session_id?: string; model?: string; }
export interface AuraAssistantDeltaEvent { event: "assistant_delta"; text: string; }
export interface AuraToolCallStartedEvent { event: "tool_call_started"; name: string; input: unknown; }
export interface AuraToolCallProgressEvent { event: "tool_call_progress"; name: string; stream: "stdout" | "stderr"; chunk: string; }
export interface AuraToolCallCompletedEvent { event: "tool_call_completed"; name: string; output: unknown; error: string | null; }
export interface AuraPermissionRequestEvent {
  event: "permission_request";
  id: string;
  tool: string;
  args: unknown;
  rule_hint: string;
  is_destructive: boolean;
}
export interface AuraFinalEvent        { event: "final"; message: string; reason: "natural" | "max_turns" | "aborted"; }
export interface AuraErrorEvent        { event: "error"; message: string; }
export interface AuraUnknownEvent      { event: "unknown"; type: string; }
export interface AuraExitedEvent       { event: "exited"; }
export interface AuraDisconnectedEvent { event: "disconnected"; }
export interface AuraStderrEvent       { event: "stderr"; line: string; }
export interface AuraRawEvent          { event: "raw"; line: string; }

export type AuraEvent =
  | AuraReadyEvent
  | AuraAssistantDeltaEvent
  | AuraToolCallStartedEvent
  | AuraToolCallProgressEvent
  | AuraToolCallCompletedEvent
  | AuraPermissionRequestEvent
  | AuraFinalEvent
  | AuraErrorEvent
  | AuraUnknownEvent
  | AuraExitedEvent
  | AuraDisconnectedEvent
  | AuraStderrEvent
  | AuraRawEvent;

/** Discriminated union of every message kind that can appear in the conversation. */
export type Message =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string; streaming: boolean; reason?: string }
  | { kind: "tool"; id: string; name: string; args: unknown; completed: boolean; error?: string | null }
  | { kind: "error"; id: string; message: string };

/** A single in-flight permission request from the Python loop. */
export interface PendingPermission {
  id: string;
  tool: string;
  args: unknown;
  ruleHint: string;
  isDestructive: boolean;
}

/** Status bar state. */
export interface Status {
  text: string;
  kind: "ready" | "thinking" | "error" | "off";
}
