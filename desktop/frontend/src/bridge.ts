/**
 * Tauri IPC bridge for Aura desktop.
 *
 * Thin wrapper around @tauri-apps/api — isolates all Tauri calls so the rest
 * of the codebase can be tested without a Tauri runtime.
 */

import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { AuraEvent } from "./types";

/**
 * Subscribe to the "aura-event" Tauri event stream.
 *
 * Returns a synchronous unsubscribe function — call it on effect cleanup.
 * Safe under React 19 StrictMode (mount → unmount → re-mount): the
 * cancellation flag handles the case where listen() hasn't resolved yet
 * when the cleanup fires.
 */
export function subscribe(handler: (ev: AuraEvent) => void): () => void {
  let unlisten: (() => void) | null = null;
  let cancelled = false;

  const listenPromise = listen<AuraEvent>("aura-event", (msg) => {
    handler(msg.payload);
  });

  listenPromise.then((fn) => {
    if (cancelled) {
      fn(); // immediately unlisten — effect already cleaned up
    } else {
      unlisten = fn;
    }
  }).catch((err) => {
    console.error("[aura bridge] listen error", err);
  });

  return () => {
    cancelled = true;
    if (unlisten) unlisten();
  };
}

/** Send a user prompt to the Rust command handler. */
export async function sendPrompt(text: string): Promise<void> {
  await invoke("send_prompt", { text });
}

/** Respond to a pending permission request. */
export async function sendPermissionResponse(
  id: string,
  choice: "accept" | "always" | "deny",
  feedback: string,
): Promise<void> {
  await invoke("send_permission_response", { id, choice, feedback });
}
