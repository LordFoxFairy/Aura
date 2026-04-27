import type { Message } from "../types";

interface Props {
  msg: Message;
}

export default function MessageBubble({ msg }: Props): React.ReactElement | null {
  switch (msg.kind) {
    case "user":
      return <div className="bubble user">{msg.text}</div>;

    case "assistant":
      return (
        <div className="bubble assistant">
          {msg.text}
          {!msg.streaming && msg.reason && msg.reason !== "natural" ? (
            <span className="reason-tag"> [{msg.reason}]</span>
          ) : null}
        </div>
      );

    case "error":
      return <div className="bubble error">error: {msg.message}</div>;

    case "tool":
      // Tool messages are rendered by ToolCard, not MessageBubble.
      return null;

    default: {
      // Exhaustive check — TypeScript will error if a new kind is added
      // without updating this switch.
      const _never: never = msg;
      console.debug("[aura MessageBubble] unhandled kind", _never);
      return null;
    }
  }
}
