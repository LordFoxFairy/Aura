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

export function formatToolOutput(output: unknown): string {
  // String passthrough preserves error messages / pre-formatted text;
  // structured values render as pretty JSON for human reading.
  if (output === undefined || output === null) return "";
  if (typeof output === "string") return output;
  try {
    return JSON.stringify(output, null, 2);
  } catch {
    return String(output);
  }
}

export default function ToolCard({ msg }: Props): React.ReactElement {
  const isError = msg.completed && msg.content?.error === true;
  const statusClass = !msg.completed
    ? "tool-card__status--running"
    : isError
      ? "tool-card__status--err"
      : "tool-card__status--ok";

  const statusText = !msg.completed
    ? "running…"
    : isError
      ? "error"
      : "ok";
  const progressText = formatToolProgress(msg.progress);
  const outputText = formatToolOutput(msg.content?.output);
  const cardClass = isError ? "tool-card tool-card--error" : "tool-card";

  return (
    <div className={cardClass}>
      <header className="tool-card__head">
        <span className="tool-card__label">{isError ? "⊘" : "tool"}</span>
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
