import type { Message } from "../types";

interface Props {
  msg: Extract<Message, { kind: "tool" }>;
}

export default function ToolCard({ msg }: Props): React.ReactElement {
  const statusClass = !msg.completed
    ? "tool-card__status--running"
    : msg.error
      ? "tool-card__status--err"
      : "tool-card__status--ok";

  const statusText = !msg.completed
    ? "running…"
    : msg.error
      ? "error"
      : "ok";

  return (
    <div className="tool-card">
      <header className="tool-card__head">
        <span className="tool-card__label">tool</span>
        <span className="tool-card__name">{msg.name}</span>
        <span className={`tool-card__status ${statusClass}`}>{statusText}</span>
      </header>
      <pre className="tool-card__args">{JSON.stringify(msg.args, null, 2)}</pre>
    </div>
  );
}
