import type { Message } from "../types";

interface Props {
  msg: Message;
}

export default function MessageBubble({ msg }: Props): React.ReactElement | null {
  switch (msg.kind) {
    case "user":
      return (
        <div className="turn turn--user">
          <div className="turn__speaker">
            <span className="speaker__label">You</span>
            <span className="speaker__rule" aria-hidden="true">&#x2014;&#x2014;</span>
          </div>
          <div className="turn__body">{msg.text}</div>
        </div>
      );

    case "assistant":
      return (
        <div className="turn turn--assistant">
          <div className="turn__speaker">
            <span className="speaker__label">Aura</span>
            <span className="speaker__rule" aria-hidden="true">&#x2014;&#x2014;</span>
          </div>
          <div className="turn__body">
            {msg.text}
            {msg.streaming && (
              <span className="streaming-caret" aria-hidden="true">&#x258C;</span>
            )}
          </div>
          {!msg.streaming && msg.reason && msg.reason !== "natural" && (
            <div className="turn__reason">[{msg.reason}]</div>
          )}
        </div>
      );

    case "error":
      return (
        <div className="turn turn--error">
          <div className="turn__speaker">
            <span className="speaker__label">Error</span>
            <span className="speaker__rule" aria-hidden="true">&#x2014;&#x2014;</span>
          </div>
          <div className="turn__body">{msg.message}</div>
        </div>
      );

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
