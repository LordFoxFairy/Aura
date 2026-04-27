import { useAuraStore } from "../store";
import * as bridge from "../bridge";
import Composer from "./Composer";

interface Pill {
  icon: string;
  label: string;
  prompt: string;
}

/** Quick-launch pills below the hero composer — verb-form, single-line. */
const PILLS: Pill[] = [
  {
    icon: "&#x2295;",
    label: "Summarize codebase",
    prompt:
      "Summarize this codebase — give a high-level walkthrough of the structure, key entry points, and conventions.",
  },
  {
    icon: "&#x2299;",
    label: "Recent changes",
    prompt:
      "Show recent git activity — last 10 commits, which files changed, and any notable patterns.",
  },
  {
    icon: "&#x25C7;",
    label: "Run tests",
    prompt:
      "Discover the test runner used by this project and execute the full test suite, then summarize the results.",
  },
  {
    icon: "&#x25B3;",
    label: "Fix lint",
    prompt:
      "Find lint errors in the project, group them by file, and apply auto-fixes where safe.",
  },
  {
    icon: "&#x25BD;",
    label: "Explain a function",
    prompt:
      "Pick a complex function in this codebase and walk me through how it works, including its callers and edge cases.",
  },
];

function firePill(prompt: string): void {
  useAuraStore.getState().appendUserMessage(prompt);
  useAuraStore.getState().setStatus("thinking…", "thinking");
  bridge.sendPrompt(prompt).catch((e: unknown) => {
    useAuraStore.getState().appendError(String(e));
    useAuraStore.getState().setStatus("ready", "ready");
  });
}

export default function WelcomeScreen(): React.ReactElement {
  return (
    <div className="welcome">
      <div className="welcome__inner">
        <h1 className="welcome__hero">How can I help today?</h1>

        <div className="welcome__composer">
          <Composer variant="hero" autoFocus />
        </div>

        <div className="welcome__pills" role="list">
          {PILLS.map((pill) => (
            <button
              key={pill.label}
              className="pill"
              type="button"
              role="listitem"
              onClick={() => { firePill(pill.prompt); }}
            >
              <span
                className="pill__icon"
                aria-hidden="true"
                dangerouslySetInnerHTML={{ __html: pill.icon }}
              />
              <span className="pill__label">{pill.label}</span>
            </button>
          ))}
        </div>

        <p className="welcome__hint">
          Aura runs locally and keeps a clean conversation history in the sidebar.
        </p>
      </div>
    </div>
  );
}
