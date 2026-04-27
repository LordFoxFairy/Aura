import { useState } from "react";
import { useAuraStore } from "../store";
import * as bridge from "../bridge";

export default function Composer(): React.ReactElement {
  const [text, setText] = useState<string>("");

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
      <textarea
        id="input"
        placeholder="Type a prompt and press Enter (Shift+Enter for newline)…"
        rows={3}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void send();
          }
        }}
      />
      <button id="send" onClick={() => void send()}>
        Send
      </button>
    </footer>
  );
}
