import type { Message } from "../types";

interface Props {
  msg: Extract<Message, { kind: "tool" }>;
}

export default function ToolCard({ msg }: Props): React.ReactElement {
  return (
    <div className={`tool-card${msg.completed ? " completed" : ""}`}>
      <div className="tool-head">🔧 {msg.name}</div>
      <pre className="tool-args">{JSON.stringify(msg.args, null, 2)}</pre>
    </div>
  );
}
