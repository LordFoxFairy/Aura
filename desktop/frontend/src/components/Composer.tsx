import { useState } from "react";
import { useAuraStore } from "../store";
import * as bridge from "../bridge";

export default function Composer(): React.ReactElement {
  const [text, setText] = useState<string>("");
  const [focused, setFocused] = useState<boolean>(false);

  const caretBlinks = focused && text.length === 0;

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

  return (
    <footer className="composer">
      <div className="composer__prompt">
        <span
          className={`composer__caret${caretBlinks ? " composer__caret--blink" : ""}`}
          aria-hidden="true"
        >
          &#x258C;
        </span>
        <textarea
          className="composer__textarea"
          placeholder="type a prompt — Enter to send, Shift+Enter for newline"
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
      </div>
      <button
        className="composer__send"
        onClick={() => void send()}
        disabled={text.trim().length === 0}
      >
        &#x25B6; SEND
      </button>
    </footer>
  );
}
