import { useState, useRef, useEffect } from "react";
import { useAuraStore } from "../store";
import * as bridge from "../bridge";

interface Props {
  /** "footer" — sticky at bottom of conversation. "hero" — large centered welcome variant. */
  variant?: "footer" | "hero";
  /** Optional autofocus on mount (used in hero / welcome state). */
  autoFocus?: boolean;
}

export default function Composer({
  variant = "footer",
  autoFocus = false,
}: Props): React.ReactElement {
  const [text, setText] = useState<string>("");
  const taRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (autoFocus && taRef.current) {
      taRef.current.focus();
    }
  }, [autoFocus]);

  // Auto-grow rows: 1 → 8 max.
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    const next = Math.min(ta.scrollHeight, 220);
    ta.style.height = `${next}px`;
  }, [text]);

  async function send(): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) return;
    setText("");
    useAuraStore.getState().appendUserMessage(trimmed);
    useAuraStore.getState().setStatus("thinking…", "thinking");
    try {
      await bridge.sendPrompt(trimmed);
    } catch (e) {
      useAuraStore.getState().appendError(String(e));
      useAuraStore.getState().setStatus("ready", "ready");
    }
  }

  const placeholder =
    variant === "hero"
      ? "Ask Aura anything — Enter to send, Shift+Enter for newline"
      : "Reply to Aura…";

  return (
    <div className={`composer composer--${variant}`}>
      <div className="composer__shell">
        <textarea
          ref={taRef}
          className="composer__input"
          placeholder={placeholder}
          rows={1}
          value={text}
          onChange={(e) => { setText(e.target.value); }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <div className="composer__actions">
          <div className="composer__actions-left">
            <button
              className="composer__icon-btn"
              type="button"
              aria-label="Attach context (coming soon)"
              title="Attach context"
              onClick={() => { console.debug("[aura] attach — Phase 2-4"); }}
            >
              {/* plus icon */}
              <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
                <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
          </div>
          <div className="composer__actions-right">
            <button
              className="composer__send"
              type="button"
              aria-label="Send"
              disabled={!text.trim()}
              onClick={() => { void send(); }}
            >
              {/* up-arrow inside a circle, evokes Manus / modern AI chat send */}
              <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
                <path
                  d="M10 16V4M10 4l-5 5M10 4l5 5"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
