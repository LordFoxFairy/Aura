import React, { useRef, useEffect } from "react";
import { useAuraStore } from "../store";
import MessageBubble from "./MessageBubble";
import ToolCard from "./ToolCard";
import { groupTurns } from "../lib/turn-groups";

export default function ConversationView(): React.ReactElement {
  const messages = useAuraStore((s) => s.messages);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll on every messages change — preserved from Phase 2-1.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const rows = groupTurns(messages);

  return (
    <div className="page" ref={scrollRef}>
      {rows.map((row) => (
        <React.Fragment key={row.id}>
          <MessageBubble msg={row.primary} />
          {row.tools.map((tool) => (
            <ToolCard key={tool.id} msg={tool} />
          ))}
        </React.Fragment>
      ))}
    </div>
  );
}
