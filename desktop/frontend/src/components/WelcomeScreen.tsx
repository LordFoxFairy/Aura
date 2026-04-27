import { useAuraStore } from "../store";
import * as bridge from "../bridge";

const EXAMPLES = [
  "summarize this codebase",
  "list lint errors in src/",
  "run the test suite",
];

export default function WelcomeScreen(): React.ReactElement {
  async function handleExample(text: string): Promise<void> {
    useAuraStore.getState().appendUserMessage(text);
    useAuraStore.getState().setStatus("thinking…", "thinking");
    try {
      await bridge.sendPrompt(text);
    } catch (e) {
      useAuraStore.getState().appendError(String(e));
      useAuraStore.getState().setStatus("ready", "ready");
    }
  }

  return (
    <div className="welcome">
      <span
        className="welcome__masthead"
        style={{ animationDelay: "0ms" }}
      >
        Aura
      </span>
      <span
        className="welcome__rule"
        aria-hidden="true"
        style={{ animationDelay: "80ms" }}
      />
      <p
        className="welcome__tagline"
        style={{ animationDelay: "80ms" }}
      >
        a quietly attentive<br />
        command-line collaborator
      </p>
      <ul className="welcome__examples" role="list">
        {EXAMPLES.map((ex, i) => (
          <li
            key={ex}
            className="welcome__example"
            role="button"
            tabIndex={0}
            onClick={() => void handleExample(ex)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                void handleExample(ex);
              }
            }}
            style={{ animationDelay: `${200 + i * 60}ms` }}
          >
            <span className="welcome__arrow" aria-hidden="true">&rarr;</span>
            <span className="welcome__example-text">{ex}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
