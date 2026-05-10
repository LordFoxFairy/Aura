import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuraStore } from "./store";
import type { Message } from "./types";

const baseState = {
  messages: [],
  status: { text: "connecting...", kind: "off" as const },
  model: "",
  permission: null,
  auraState: null,
  rightPanelOpen: false,
};

function resetStore(): void {
  useAuraStore.setState(baseState);
}

function toolMessages(): Array<Extract<Message, { kind: "tool" }>> {
  return useAuraStore
    .getState()
    .messages
    .filter((message): message is Extract<Message, { kind: "tool" }> => message.kind === "tool");
}

describe("Aura desktop store", () => {
  beforeEach(() => {
    let nextId = 0;
    vi.restoreAllMocks();
    vi.spyOn(crypto, "randomUUID").mockImplementation(() => {
      const suffix = String(++nextId).padStart(12, "0");
      return `00000000-0000-4000-8000-${suffix}`;
    });
    resetStore();
  });

  it("preserves tool content when a tool call completes", () => {
    // Phase 2 Task 9: ``content: {text, error}`` replaces split
    // output/error fields.
    const store = useAuraStore.getState();

    store.appendToolCall("tc_1", "read_file", { path: "README.md" });
    store.completeToolCall("tc_1", "read_file", {
      text: '{"content":"hello"}',
      error: false,
    });

    expect(toolMessages()).toEqual([
      {
        kind: "tool",
        id: "tc_1",
        name: "read_file",
        args: { path: "README.md" },
        completed: true,
        content: { text: '{"content":"hello"}', error: false },
        progress: [],
      },
    ]);
  });

  it("preserves error flag when tool fails", () => {
    // Phase 2 Task 9: a failing tool's ``content.error`` is true so
    // the frontend can render the red banner / ⊘ glyph.
    const store = useAuraStore.getState();

    store.appendToolCall("tc_err", "bash", { command: "rm -rf /" });
    store.completeToolCall("tc_err", "bash", {
      text: "permission denied",
      error: true,
    });

    expect(toolMessages()[0]?.content).toEqual({
      text: "permission denied",
      error: true,
    });
  });

  it("keeps completed tools immutable when late progress arrives", () => {
    const store = useAuraStore.getState();

    store.appendToolCall("tc_1", "bash", { command: "printf hi" });
    store.completeToolCall("tc_1", "bash", { text: '{"stdout":"hi"}', error: false });
    store.appendToolProgress("tc_1", "bash", "stdout", "late");

    expect(toolMessages()[0]?.progress).toEqual([]);
  });

  it("matches id-less same-name tool events as a stack", () => {
    const store = useAuraStore.getState();

    store.appendToolCall(undefined, "bash", { command: "one" });
    store.appendToolCall(undefined, "bash", { command: "two" });
    store.appendToolProgress(undefined, "bash", "stdout", "two running");
    store.completeToolCall(undefined, "bash", { text: '{"stdout":"two"}', error: false });
    store.completeToolCall(undefined, "bash", { text: '{"stdout":"one"}', error: false });

    expect(toolMessages()).toMatchObject([
      {
        name: "bash",
        args: { command: "one" },
        completed: true,
        content: { text: '{"stdout":"one"}', error: false },
        progress: [],
      },
      {
        name: "bash",
        args: { command: "two" },
        completed: true,
        content: { text: '{"stdout":"two"}', error: false },
        progress: [{ stream: "stdout", chunk: "two running" }],
      },
    ]);
  });

  it("does not match duplicate wire ids across different tool names", () => {
    const store = useAuraStore.getState();

    store.appendToolCall("dup", "bash", { command: "printf hi" });
    store.appendToolCall("dup", "read_file", { path: "README.md" });
    store.completeToolCall("dup", "bash", { text: '{"stdout":"hi"}', error: false });

    expect(toolMessages()).toMatchObject([
      {
        id: "dup",
        name: "bash",
        completed: true,
        content: { text: '{"stdout":"hi"}', error: false },
      },
      {
        id: "dup",
        name: "read_file",
        completed: false,
      },
    ]);
  });

  it("treats an empty wire id as missing", () => {
    const store = useAuraStore.getState();

    store.appendToolCall("", "bash", { command: "printf hi" });
    store.completeToolCall("", "bash", { text: '{"stdout":"hi"}', error: false });

    expect(toolMessages()[0]).toMatchObject({
      id: "00000000-0000-4000-8000-000000000001",
      name: "bash",
      completed: true,
      content: { text: '{"stdout":"hi"}', error: false },
    });
  });

  it("records permission audits as conversation messages", () => {
    useAuraStore.getState().appendPermissionAudit("bash", "auto-allowed");

    expect(useAuraStore.getState().messages).toEqual([
      {
        kind: "audit",
        id: "00000000-0000-4000-8000-000000000001",
        tool: "bash",
        text: "auto-allowed",
      },
    ]);
  });
});
