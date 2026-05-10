import { useAuraStore } from "../store";

export default function TopNav(): React.ReactElement {
  const messages = useAuraStore((s) => s.messages);
  const model = useAuraStore((s) => s.model);
  const auraState = useAuraStore((s) => s.auraState);
  const rightPanelOpen = useAuraStore((s) => s.rightPanelOpen);
  const toggleRightPanel = useAuraStore((s) => s.toggleRightPanel);

  const mode = auraState?.mode ?? "default";

  // Fix 7: derive title from first user message, truncated to 50 chars.
  const firstUser = messages.find((m) => m.kind === "user");
  const isPlaceholder = !firstUser;
  const rawTitle = firstUser
    ? (firstUser as Extract<typeof firstUser, { kind: "user" }>).text
    : "Untitled conversation";
  const title =
    rawTitle.length > 50 ? rawTitle.slice(0, 50) + "…" : rawTitle;

  return (
    <header className="topnav">
      <div className="topnav__left">
        {/* Fix 1: placeholder class when no user messages */}
        <h1
          className={
            isPlaceholder
              ? "topnav__title topnav__title--placeholder"
              : "topnav__title"
          }
        >
          {title}
        </h1>
      </div>
      <div className="topnav__right">
        {/* Fix 1: "Connecting…" pill with pulsing dot when no model loaded */}
        {model ? (
          <div className="topnav__model" data-mode={mode}>
            <span className="model-dot" />
            <span className="model-name">{model}</span>
          </div>
        ) : (
          <div className="topnav__model topnav__model--connecting" data-mode="off">
            <span className="model-dot model-dot--pulsing" />
            <span className="model-name">Connecting…</span>
          </div>
        )}
        {mode !== "default" && (
          <span className="mode-pill" data-mode={mode}>{mode}</span>
        )}
        {/* Fix 2: inline SVG panel-toggle icon replacing Unicode ⌶ */}
        <button
          className="topnav__toggle"
          onClick={toggleRightPanel}
          aria-label="Toggle context panel"
          aria-pressed={rightPanelOpen}
        >
          <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
            <rect x="1.5" y="2.5" width="13" height="11" rx="2" stroke="currentColor" strokeWidth="1.3" fill="none" />
            <line x1="10" y1="2.5" x2="10" y2="13.5" stroke="currentColor" strokeWidth="1.3" />
          </svg>
        </button>
      </div>
    </header>
  );
}
