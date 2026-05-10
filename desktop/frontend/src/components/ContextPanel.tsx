import { useAuraStore } from "../store";
import ToolCard from "./ToolCard";
import type { Message } from "../types";

/** Compact number formatter: <1k → raw, <1M → "1.2k", else "1.2M". */
function humanize(n: number): string {
  if (n < 1_000) return String(n);
  if (n < 1_000_000) return (n / 1_000).toFixed(1) + "k";
  return (n / 1_000_000).toFixed(1) + "M";
}

/** Last segment of a Unix/Windows path; falls back to the full path or "?" for empty. */
function basename(p: string): string {
  if (!p) return "?";
  const parts = p.replace(/\\/g, "/").split("/").filter(Boolean);
  return parts.length > 0 ? (parts[parts.length - 1] ?? "?") : "/";
}

export function findActiveTool(messages: Message[]): Extract<Message, { kind: "tool" }> | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.kind === "tool" && !m.completed) {
      return m as Extract<Message, { kind: "tool" }>;
    }
  }
  return null;
}

export function findLatestTool(messages: Message[]): Extract<Message, { kind: "tool" }> | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.kind === "tool") {
      return m as Extract<Message, { kind: "tool" }>;
    }
  }
  return null;
}

export default function ContextPanel(): React.ReactElement {
  const rightPanelOpen = useAuraStore((s) => s.rightPanelOpen);
  const auraState = useAuraStore((s) => s.auraState);
  const messages = useAuraStore((s) => s.messages);

  const activeTool = findActiveTool(messages);
  const visibleTool = activeTool ?? findLatestTool(messages);

  // Token gauge calc
  let live = 0;
  let windowSize = 0;
  let pct = 0;
  if (auraState) {
    live = auraState.pinned + auraState.tokens.last_input;
    windowSize = auraState.window;
    pct = windowSize > 0 ? Math.min(100, Math.max(0, Math.round((live * 100) / windowSize))) : 0;
  }

  return (
    <aside className="context" data-open={String(rightPanelOpen)}>
      {/* Live tool section */}
      <div className="context__section">
        <h4 className="context__label">{activeTool ? "Live tool" : "Last tool"}</h4>
        {visibleTool
          ? <ToolCard msg={visibleTool} />
          : <div className="empty-hint">No tool running</div>
        }
      </div>

      {/* Session section */}
      <div className="context__section">
        <h4 className="context__label">Session</h4>
        {auraState ? (
          <dl className="session-meta">
            <dt>Model</dt>
            <dd>{auraState.model || "—"}</dd>
            <dt>Mode</dt>
            <dd>{auraState.mode}</dd>
            <dt>Context</dt>
            <dd>{humanize(live)}/{humanize(windowSize)} ({pct}%)</dd>
            <dt>Cwd</dt>
            <dd>{basename(auraState.cwd)}</dd>
            <dt>Last turn</dt>
            <dd>{auraState.last_turn_seconds > 0 ? `${auraState.last_turn_seconds.toFixed(1)}s` : "—"}</dd>
          </dl>
        ) : (
          <div className="empty-hint">Warming up…</div>
        )}
      </div>

      {/* Token budget section */}
      <div className="context__section">
        <h4 className="context__label">Token budget</h4>
        <div className="gauge">
          <div className="gauge__bar">
            <div
              className={`gauge__fill${pct >= 90 ? " gauge__fill--critical" : ""}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="gauge__label">
            {auraState
              ? `${humanize(live)} of ${humanize(windowSize)}`
              : "— of —"
            }
          </div>
        </div>
      </div>
    </aside>
  );
}
