import { useAuraStore } from "../store";

export default function TopNav(): React.ReactElement {
  const messages = useAuraStore((s) => s.messages);
  const model = useAuraStore((s) => s.model);
  const auraState = useAuraStore((s) => s.auraState);
  const rightPanelOpen = useAuraStore((s) => s.rightPanelOpen);
  const toggleRightPanel = useAuraStore((s) => s.toggleRightPanel);

  const mode = auraState?.mode ?? "default";

  const firstUser = messages.find((m) => m.kind === "user");
  const title = firstUser
    ? (firstUser as Extract<typeof firstUser, { kind: "user" }>).text.slice(0, 60)
    : "Untitled conversation";

  return (
    <header className="topnav">
      <div className="topnav__left">
        <h1 className="topnav__title">{title}</h1>
      </div>
      <div className="topnav__right">
        <div className="topnav__model" data-mode={mode}>
          <span className="model-dot" />
          <span className="model-name">{model || "no model"}</span>
        </div>
        {mode !== "default" && (
          <span className="mode-pill" data-mode={mode}>{mode}</span>
        )}
        <button
          className="topnav__toggle"
          onClick={toggleRightPanel}
          aria-label="Toggle context panel"
          aria-pressed={rightPanelOpen}
        >
          &#x2336;
        </button>
      </div>
    </header>
  );
}
