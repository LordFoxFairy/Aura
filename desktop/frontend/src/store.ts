/**
 * Zustand store for Aura desktop.
 *
 * Pure data layer — no React components, no DOM access, no side effects.
 * All reducers are immutable: they call set(...) and return changed slices only;
 * zustand merges shallow.
 */

import { create } from "zustand";
import type { Message, PendingPermission, Status } from "./types";

export interface AuraStore {
  messages: Message[];
  status: Status;
  model: string;
  permission: PendingPermission | null;

  appendUserMessage(text: string): void;
  appendAssistantDelta(text: string): void;
  finalizeAssistant(reason: string): void;
  appendToolCall(name: string, args: unknown): void;
  completeToolCall(name: string, error: string | null): void;
  showPermission(req: PendingPermission): void;
  hidePermission(): void;
  setStatus(text: string, kind: Status["kind"]): void;
  appendError(message: string): void;
  setDisconnected(): void;
  setReady(model: string): void;
}

export const useAuraStore = create<AuraStore>((set) => ({
  // ── Initial state ──────────────────────────────────────────────────────────
  messages: [],
  status: { text: "connecting…", kind: "off" },
  model: "",
  permission: null,

  // ── Actions ────────────────────────────────────────────────────────────────

  appendUserMessage(text: string): void {
    set((state) => ({
      messages: [
        ...state.messages,
        { kind: "user", id: crypto.randomUUID(), text },
      ],
    }));
  },

  appendAssistantDelta(text: string): void {
    set((state) => {
      const msgs = state.messages;
      const last = msgs.length > 0 ? msgs[msgs.length - 1] : undefined;
      if (last !== undefined && last.kind === "assistant" && last.streaming) {
        // Extend the existing streaming bubble in place (immutable).
        return {
          messages: [
            ...msgs.slice(0, msgs.length - 1),
            { ...last, text: last.text + text },
          ],
        };
      }
      // No streaming bubble yet — create one.
      return {
        messages: [
          ...msgs,
          { kind: "assistant", id: crypto.randomUUID(), text, streaming: true },
        ],
      };
    });
  },

  finalizeAssistant(reason: string): void {
    set((state) => {
      const msgs = state.messages;
      // Walk from end to find last streaming assistant (no extra allocation).
      for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i];
        if (m.kind === "assistant" && m.streaming) {
          const finalized: Extract<Message, { kind: "assistant" }> = {
            ...(m as Extract<Message, { kind: "assistant" }>),
            streaming: false,
            ...(reason !== "natural" && reason !== "" ? { reason } : {}),
          };
          return {
            messages: [
              ...msgs.slice(0, i),
              finalized,
              ...msgs.slice(i + 1),
            ],
          };
        }
      }
      return {}; // no-op — nothing to finalize
    });
  },

  appendToolCall(name: string, args: unknown): void {
    set((state) => {
      // Inline-finalize any open streaming assistant bubble, then push the
      // tool row — all in one dispatch so Tauri event callbacks never see an
      // intermediate state between the two mutations.
      let msgs = state.messages;
      for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i];
        if (m.kind === "assistant" && m.streaming) {
          const finalized: Extract<Message, { kind: "assistant" }> = {
            ...(m as Extract<Message, { kind: "assistant" }>),
            streaming: false,
          };
          msgs = [...msgs.slice(0, i), finalized, ...msgs.slice(i + 1)];
          break;
        }
      }
      return {
        messages: [
          ...msgs,
          { kind: "tool", id: crypto.randomUUID(), name, args, completed: false },
        ],
      };
    });
  },

  // NOTE: match is by tool name because the Python wire (headless.py
  // _event_to_dict) doesn't include an `id` on tool_call_* events. This is
  // safe today because the agent loop is serial (one tool at a time). If
  // parallel tool calls land in v0.15+, headless.py must add an `id` field
  // and this reducer should match by id instead (Python schema change, out of
  // scope here).
  completeToolCall(name: string, error: string | null): void {
    set((state) => {
      const msgs = state.messages;
      // Scan from end to find the last incomplete tool call with this name.
      for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i];
        if (m.kind === "tool" && m.name === name && !m.completed) {
          const updated: Extract<Message, { kind: "tool" }> = {
            ...(m as Extract<Message, { kind: "tool" }>),
            completed: true,
            error,
          };
          return {
            messages: [
              ...msgs.slice(0, i),
              updated,
              ...msgs.slice(i + 1),
            ],
          };
        }
      }
      return {}; // no-op — no matching incomplete tool call
    });
  },

  showPermission(req: PendingPermission): void {
    set({ permission: req });
  },

  hidePermission(): void {
    set({ permission: null });
  },

  setStatus(text: string, kind: Status["kind"]): void {
    set({ status: { text, kind } });
  },

  appendError(message: string): void {
    set((state) => ({
      messages: [
        ...state.messages,
        { kind: "error", id: crypto.randomUUID(), message },
      ],
    }));
  },

  setDisconnected(): void {
    set({ status: { text: "aura exited", kind: "off" } });
  },

  setReady(model: string): void {
    set({ model, status: { text: `ready · ${model}`, kind: "ready" } });
  },
}));
