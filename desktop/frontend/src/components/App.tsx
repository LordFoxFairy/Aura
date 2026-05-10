import { useEffect } from "react";
import { useAuraStore } from "../store";
import * as bridge from "../bridge";
import type { AuraEvent } from "../types";
import ConversationView from "./ConversationView";
import WelcomeScreen from "./WelcomeScreen";
import Composer from "./Composer";
import PermissionModal from "./PermissionModal";
import Sidebar from "./Sidebar";
import TopNav from "./TopNav";
import ContextPanel from "./ContextPanel";

function dispatch(ev: AuraEvent): void {
  // Use getState() — stable reference, no render dependency in this dispatcher.
  const {
    setReady,
    appendAssistantDelta,
    appendToolCall,
    appendToolProgress,
    completeToolCall,
    appendPermissionAudit,
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
      appendToolCall(ev.id, ev.name, ev.input);
      break;

    case "tool_call_progress":
      appendToolProgress(ev.id, ev.name, ev.stream, ev.chunk);
      break;

    case "tool_call_completed":
      completeToolCall(ev.id, ev.name, ev.content);
      break;

    case "permission_audit":
      appendPermissionAudit(ev.tool, ev.text);
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
  const rightPanelOpen = useAuraStore((s) => s.rightPanelOpen);
  const isWelcome = messages.filter((m) => m.kind !== "tool").length === 0;

  useEffect(() => {
    const unsubscribe = bridge.subscribe(dispatch);
    return unsubscribe;
  }, []);

  return (
    <div
      id="app"
      data-right-panel-closed={String(!rightPanelOpen)}
      data-welcome={String(isWelcome)}
    >
      <Sidebar />

      <div className="main-column">
        <TopNav />
        <div className="conversation">
          {isWelcome ? <WelcomeScreen /> : <ConversationView />}
        </div>
        {/* Bottom composer only in conversation mode — welcome embeds its own hero composer. */}
        {!isWelcome && <Composer variant="footer" />}
      </div>

      <ContextPanel />
      <PermissionModal />
    </div>
  );
}
