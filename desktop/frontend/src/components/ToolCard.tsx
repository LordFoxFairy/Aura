import type { Message } from "../types";

interface Props {
  msg: Extract<Message, { kind: "tool" }>;
}

export function formatToolProgress(
  progress: Array<{ stream: "stdout" | "stderr"; chunk: string }>,
): string {
  return progress
    .map((entry) => `${entry.stream}> ${entry.chunk.replace(/\n$/, "")}`)
    .join("\n");
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
  const progressText = formatToolProgress(msg.progress);
  const outputText = msg.output === undefined
    ? ""
    : typeof msg.output === "string"
      ? msg.output
      : JSON.stringify(msg.output, null, 2);

  return (
    <div className="tool-card">
      <header className="tool-card__head">
        <span className="tool-card__label">tool</span>
        <span className="tool-card__name">{msg.name}</span>
        <span className={`tool-card__status ${statusClass}`}>{statusText}</span>
      </header>
      <pre className="tool-card__args">{JSON.stringify(msg.args, null, 2)}</pre>
      {progressText !== "" && (
        <pre className="tool-card__progress">{progressText}</pre>
      )}
      {msg.completed && outputText !== "" && (
        <pre className="tool-card__output">{outputText}</pre>
      )}
    </div>
  );
}
