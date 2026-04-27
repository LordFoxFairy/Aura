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

      <div className="sidebar__search">
        <input placeholder="Search conversations…" type="search" />
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
        <button
          className="sidebar__link"
          onClick={() => { console.log("[aura] Settings clicked"); }}
        >
          &#x2699; Settings
        </button>
        <button
          className="sidebar__link"
          onClick={() => { console.log("[aura] Help clicked"); }}
        >
          ? Help
        </button>
      </div>
    </aside>
  );
}
