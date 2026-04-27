import { useAuraStore } from "../store";
import * as bridge from "../bridge";

export default function PermissionModal(): React.ReactElement | null {
  const permission = useAuraStore((s) => s.permission);

  if (!permission) return null;

  const handle = async (choice: "accept" | "always" | "deny"): Promise<void> => {
    // Read live permission from store — avoids stale closure if a second
    // permission_request arrives while the modal is already open.
    const current = useAuraStore.getState().permission;
    if (!current) return;
    try {
      await bridge.sendPermissionResponse(current.id, choice, "");
    } catch (e) {
      useAuraStore.getState().appendError(`permission response failed: ${String(e)}`);
    }
    useAuraStore.getState().hidePermission();
  };

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div
        className="modal"
        data-destructive={String(permission.isDestructive)}
      >
        <header className="modal__head">
          <span className="modal__icon">!</span>
          <h2 className="modal__title">Aura wants to run a tool</h2>
        </header>

        <dl className="modal__body">
          <dt>Tool</dt>
          <dd className="mono">{permission.tool}</dd>
          <dt>Arguments</dt>
          <dd className="mono">
            <pre>{JSON.stringify(permission.args, null, 2)}</pre>
          </dd>
          {permission.ruleHint && (
            <>
              <dt>If you choose &ldquo;Always&rdquo;</dt>
              <dd className="mono">{permission.ruleHint}</dd>
            </>
          )}
        </dl>

        <footer className="modal__actions">
          <button className="btn btn--ghost" onClick={() => { void handle("deny"); }}>
            No
          </button>
          <button className="btn btn--ghost" onClick={() => { void handle("accept"); }}>
            Yes, once
          </button>
          <button className="btn btn--primary" onClick={() => { void handle("always"); }}>
            Yes, always
          </button>
        </footer>
      </div>
    </div>
  );
}
