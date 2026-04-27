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
    <div className="overlay" id="permission-overlay">
      <div
        className="permission-modal"
        data-destructive={String(permission.isDestructive)}
      >
        <div className="permission-head">
          <span className="permission-icon" aria-hidden="true">⚠</span>
          <span>Aura wants to run a tool</span>
        </div>
        <div className="permission-tool">{permission.tool}</div>
        <pre className="permission-args">
          {JSON.stringify(permission.args, null, 2)}
        </pre>
        <div className="permission-actions">
          <button className="btn-deny" onClick={() => void handle("deny")}>
            No
          </button>
          <button className="btn-once" onClick={() => void handle("accept")}>
            Yes, once
          </button>
          <button className="btn-always" onClick={() => void handle("always")}>
            Yes, always
          </button>
        </div>
        <div className="permission-hint">
          {permission.ruleHint
            ? `"Always" installs rule: ${permission.ruleHint}`
            : ""}
        </div>
      </div>
    </div>
  );
}
