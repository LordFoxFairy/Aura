import { useEffect } from "react";
import { useAuraStore } from "../store";
import * as bridge from "../bridge";
import type { AuraEvent } from "../types";
import ConversationView from "./ConversationView";
import WelcomeScreen from "./WelcomeScreen";
import Composer from "./Composer";
import PermissionModal from "./PermissionModal";
import StatusBar from "./StatusBar";

function dispatch(ev: AuraEvent): void {
  // Use getState() — stable reference, no render dependency in this dispatcher.
  const {
    setReady,
    appendAssistantDelta,
    appendToolCall,
    completeToolCall,
    showPermission,
    finalizeAssistant,
    appendError,
    setStatus,
    setDisconnected,
    applyAuraState,
  } = useAuraStore.getState();

  switch (ev.event) {
    case "ready":
      setReady(ev.model ?? "");
      break;

    case "assistant_delta":
      appendAssistantDelta(ev.text);
      break;

    case "tool_call_started":
      appendToolCall(ev.name, ev.input);
      break;

    case "tool_call_progress":
      // No-op for Phase 2-1. Phase 2-4 will use live bash chunks here.
      console.debug("[aura tool_call_progress]", ev);
      break;

    case "tool_call_completed":
      completeToolCall(ev.name, ev.error);
      break;

    case "permission_request":
      // Runtime sanity guard: wire could theoretically send malformed payload.
      if (ev.id && ev.tool) {
        showPermission({
          id: ev.id,
          tool: ev.tool,
          args: ev.args ?? {},
          ruleHint: ev.rule_hint ?? "",
          // Defensive default true (matches spec): if wire omits the field,
          // assume destructive — louder visual treatment, fail-safe.
          isDestructive: ev.is_destructive !== false,
        });
      }
      break;

    case "final":
      finalizeAssistant(ev.reason);
      setStatus("ready", "ready");
      break;

    case "error":
      appendError(ev.message);
      setStatus("error", "error");
      break;

    case "unknown":
      console.debug("[aura unknown]", ev);
      break;

    case "exited":
      setDisconnected();
      break;

    case "disconnected":
      setDisconnected();
      break;

    case "stderr":
      console.warn("[aura stderr]", ev.line);
      break;

    case "raw":
      console.warn("[aura raw]", ev.line);
      break;

    case "aura_state":
      applyAuraState({
        model: ev.model,
        mode: ev.mode,
        cwd: ev.cwd,
        tokens: ev.tokens,
        pinned: ev.pinned,
        window: ev.window,
        last_turn_seconds: ev.last_turn_seconds,
      });
      break;

    default: {
      // Forward-compat: future Python versions may emit new event types.
      const forward = ev as { event: string; [k: string]: unknown };
      console.debug("[aura event]", forward.event, forward);
      break;
    }
  }
}

export default function App(): React.ReactElement {
  const messages = useAuraStore((s) => s.messages);

  useEffect(() => {
    const unsubscribe = bridge.subscribe(dispatch);
    return unsubscribe;
  }, []);

  return (
    <div id="app">
      <header className="topbar">
        <span className="brand">Aura</span>
        <span className="meta">editorial terminal</span>
      </header>
      {messages.length === 0 ? (
        // Welcome state: render a .page grid with WelcomeScreen inside it.
        <div className="page">
          <WelcomeScreen />
        </div>
      ) : (
        // Conversation state: ConversationView renders its own .page grid.
        <ConversationView />
      )}
      <StatusBar />
      <Composer />
      <PermissionModal />
    </div>
  );
}
