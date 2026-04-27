import type { Message } from "../types";

interface Props {
  msg: Message;
}

export default function MessageBubble({ msg }: Props): React.ReactElement | null {
  switch (msg.kind) {
    case "user":
      return (
        <div className="msg msg--user">
          <div className="msg__bubble">{msg.text}</div>
        </div>
      );

    case "assistant":
      return (
        <div className="msg msg--assistant">
          <div className="msg__speaker">Aura</div>
          <div className="msg__body">
            {msg.text}
            {msg.streaming && (
              <span className="streaming-caret" aria-hidden="true">&#x258D;</span>
            )}
          </div>
          {!msg.streaming && msg.reason && msg.reason !== "natural" && (
            <div className="msg__reason">stopped: {msg.reason}</div>
          )}
        </div>
      );

    case "error":
      return (
        <div className="msg msg--error">
          <div className="msg__icon">!</div>
          <div className="msg__body">{msg.message}</div>
        </div>
      );

    case "tool":
      // Tool messages render inside ContextPanel, not in ConversationView.
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
