import { describe, expect, it } from "vitest";
import { findActiveTool, findLatestTool } from "./ContextPanel";
import { formatToolProgress } from "./ToolCard";
import type { Message } from "../types";

describe("findActiveTool", () => {
  it("returns the newest unfinished tool", () => {
    const messages: Message[] = [
      {
        kind: "tool",
        id: "older",
        name: "bash",
        args: {},
        completed: false,
        progress: [],
      },
      {
        kind: "tool",
        id: "newer",
        name: "read_file",
        args: {},
        completed: false,
        progress: [],
      },
    ];

    expect(findActiveTool(messages)?.id).toBe("newer");
  });

  it("ignores completed tools", () => {
    const messages: Message[] = [
      {
        kind: "tool",
        id: "completed",
        name: "bash",
        args: {},
        completed: true,
        output: { ok: true },
        error: null,
        progress: [],
      },
    ];

    expect(findActiveTool(messages)).toBeNull();
  });

  it("keeps the newest completed tool available for last-tool rendering", () => {
    const messages: Message[] = [
      {
        kind: "tool",
        id: "completed",
        name: "read_file",
        args: {},
        completed: true,
        output: { content: "hello" },
        error: null,
        progress: [],
      },
    ];

    expect(findLatestTool(messages)?.output).toEqual({ content: "hello" });
  });

  it("formats progress chunks with stable line boundaries", () => {
    expect(formatToolProgress([
      { stream: "stdout", chunk: "one" },
      { stream: "stderr", chunk: "two\n" },
    ])).toBe("stdout> one\nstderr> two");
  });
});
