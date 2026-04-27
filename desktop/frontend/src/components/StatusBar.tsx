import { useAuraStore } from "../store";
import type { AuraStateSnapshot } from "../types";

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

function renderGauge(snap: AuraStateSnapshot): React.ReactElement {
  if (snap.window === 0) {
    return <span className="seg gauge">&#x2014; tokens</span>;
  }
  const live = snap.pinned + snap.tokens.last_input;
  const pct = Math.min(100, Math.max(0, Math.round((live * 100) / snap.window)));
  return (
    <span className="seg gauge">
      ctx {humanize(live)}/{humanize(snap.window)} {pct}%
    </span>
  );
}

export default function StatusBar(): React.ReactElement {
  const auraState = useAuraStore((s) => s.auraState);

  if (!auraState) {
    return (
      <aside className="status-bar">
        <span className="seg italic">warming&#x2026;</span>
      </aside>
    );
  }

  const { model, mode, cwd, tokens, last_turn_seconds } = auraState;

  return (
    <aside className="status-bar">
      {/* 1. Model */}
      <span className="seg model">{model || "?"}</span>

      {/* 2. Token gauge — no ASCII bar */}
      {renderGauge(auraState)}

      {/* 3. Cache read — only when non-zero */}
      {tokens.last_cache_read > 0 && (
        <span className="seg cache">+{humanize(tokens.last_cache_read)} cached</span>
      )}

      {/* 4. Mode — hidden for "default"; text color via data-mode */}
      {mode !== "default" && (
        <span className="seg mode" data-mode={mode}>{mode}</span>
      )}

      {/* 5. Cwd basename */}
      <span className="seg cwd">{basename(cwd)}</span>

      {/* 6. Last turn duration — only when non-zero */}
      {last_turn_seconds > 0 && (
        <span className="seg dur">{last_turn_seconds.toFixed(1)}s</span>
      )}
    </aside>
  );
}
