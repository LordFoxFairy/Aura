import type { Message } from "../types";

interface Props {
  msg: Extract<Message, { kind: "tool" }>;
}

export default function ToolCard({ msg }: Props): React.ReactElement {
  const hasError = Boolean(msg.error);
  const className = [
    "tool",
    msg.completed ? "tool--done" : "",
    hasError ? "tool--err" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <aside className={className}>
      <header className="tool__head">
        <span className="tool__label">tool</span>
        <span className="tool__sep" aria-hidden="true">&middot;</span>
        <span className="tool__name">{msg.name}</span>
      </header>
      <pre className="tool__args">{JSON.stringify(msg.args, null, 2)}</pre>
      {msg.completed && (
        <div className="tool__status">{hasError ? "error" : "ok"}</div>
      )}
    </aside>
  );
}
