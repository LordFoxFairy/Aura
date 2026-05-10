import { useAuraStore } from "../store";

export default function Sidebar(): React.ReactElement {
  function handleNewChat(): void {
    useAuraStore.getState().clearMessages();
  }

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="brand-mark">Aura</span>
      </div>

      <button
        className="btn btn--primary new-chat-btn"
        onClick={handleNewChat}
      >
        + New conversation
      </button>

      {/* Fix 4: magnifying-glass prefix icon inside wrapper */}
      <div className="sidebar__search">
        <span className="sidebar__search-icon" aria-hidden="true">
          <svg viewBox="0 0 16 16" width="14" height="14">
            <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.3" fill="none" />
            <line x1="10.5" y1="10.5" x2="13.5" y2="13.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>
        </span>
        <input type="search" placeholder="Search…" />
      </div>

      <nav className="sidebar__sessions" aria-label="Conversation history">
        <div className="session-group">
          <h4 className="session-group__label">Today</h4>
          <ul>
            <li className="session-item is-active">
              <span className="session-item__title">Current session</span>
              <span className="session-item__time">now</span>
            </li>
          </ul>
        </div>
      </nav>

      <div className="sidebar__footer">
        {/* Fix 3: SVG gear icon replacing ⚙ */}
        <button
          className="sidebar__link"
          onClick={() => { console.log("[aura] Settings clicked"); }}
        >
          <span className="sidebar__link-icon" aria-hidden="true">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
              <circle cx="8" cy="8" r="2.2" stroke="currentColor" strokeWidth="1.3" fill="none" />
              <path d="M8 1.5v2M8 12.5v2M14.5 8h-2M3.5 8h-2M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4M12.6 12.6l-1.4-1.4M4.8 4.8L3.4 3.4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </span>
          Settings
        </button>
        {/* Fix 3: SVG circled question-mark icon replacing ? */}
        <button
          className="sidebar__link"
          onClick={() => { console.log("[aura] Help clicked"); }}
        >
          <span className="sidebar__link-icon" aria-hidden="true">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
              <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.3" fill="none" />
              <path d="M6 6.2c0-1.2 1-2 2-2s2 .7 2 1.8c0 1.1-1 1.5-1.5 2-.4.4-.5.7-.5 1.3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" fill="none" />
              <circle cx="8" cy="11.5" r="0.7" fill="currentColor" />
            </svg>
          </span>
          Help
        </button>
      </div>
    </aside>
  );
}
