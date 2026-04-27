import { useRef, useEffect } from "react";
import { useAuraStore } from "../store";
import MessageBubble from "./MessageBubble";
import ToolCard from "./ToolCard";
import type { Message } from "../types";

export default function ConversationView(): React.ReactElement {
  const messages = useAuraStore((s) => s.messages);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <main id="conversation" ref={scrollRef}>
      {messages.map((msg: Message) =>
        msg.kind === "tool" ? (
          <ToolCard key={msg.id} msg={msg} />
        ) : (
          <MessageBubble key={msg.id} msg={msg} />
        ),
      )}
    </main>
  );
}
