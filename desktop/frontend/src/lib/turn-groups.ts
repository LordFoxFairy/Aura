/**
 * turn-groups.ts — pure helper for grouping flat message list into turn rows.
 *
 * A "TurnRow" is one conversational unit: a primary message (user/assistant/error)
 * plus any tool calls that were emitted as part of that turn.
 *
 * Tool messages that appear before any primary message are orphans — they are
 * dropped with a console.debug warning (shouldn't happen in practice).
 */

import type { Message } from "../types";

export type AssistantTurn = Extract<Message, { kind: "assistant" }>;
export type UserTurn      = Extract<Message, { kind: "user" }>;
export type ErrorTurn     = Extract<Message, { kind: "error" }>;
export type ToolTurn      = Extract<Message, { kind: "tool" }>;

export interface TurnRow {
  id: string;
  primary: UserTurn | AssistantTurn | ErrorTurn;
  tools: ToolTurn[];
}

export function groupTurns(messages: Message[]): TurnRow[] {
  const rows: TurnRow[] = [];

  for (const msg of messages) {
    if (msg.kind === "tool") {
      const last = rows.length > 0 ? rows[rows.length - 1] : undefined;
      if (!last) {
        // Orphan tool message — drop it defensively.
        console.debug("[aura groupTurns] orphan tool message, dropping:", msg);
        continue;
      }
      last.tools.push(msg as ToolTurn);
    } else {
      // user | assistant | error — start a new row
      rows.push({
        id: msg.id,
        primary: msg as UserTurn | AssistantTurn | ErrorTurn,
        tools: [],
      });
    }
  }

  return rows;
}
