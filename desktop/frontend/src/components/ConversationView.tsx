import { useRef, useEffect } from "react";
import { useAuraStore } from "../store";
import MessageBubble from "./MessageBubble";

export default function ConversationView(): React.ReactElement {
  const messages = useAuraStore((s) => s.messages);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom on every messages reference change.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const visible = messages.filter((m) => m.kind !== "tool");

  return (
    <div className="conversation__inner">
      {visible.map((msg) => (
        <MessageBubble key={msg.id} msg={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
