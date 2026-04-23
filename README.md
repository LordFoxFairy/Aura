# Aura

[![PyPI](https://img.shields.io/pypi/v/aura-agent.svg)](https://pypi.org/project/aura-agent/) [![Python Versions](https://img.shields.io/pypi/pyversions/aura-agent.svg)](https://pypi.org/project/aura-agent/) [![CI](https://github.com/LordFoxFairy/Aura/actions/workflows/ci.yml/badge.svg)](https://github.com/LordFoxFairy/Aura/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight Python agent with an explicit async loop, pluggable LLM providers, and a hook-based extension model. Built on LangChain as a client-only layer — no LangGraph, no LangChain agents, no LCEL.

## Design in one paragraph

Aura owns the agent loop. LangChain is used strictly as a uniform `BaseChatModel` client over OpenAI / Anthropic / Ollama and any OpenAI-compatible endpoint (OpenRouter, DeepSeek, self-hosted). The loop dispatches tool calls through a `ToolRegistry` of LangChain `BaseTool` subclasses that carry Aura-specific metadata (`is_read_only`, `is_destructive`, `is_concurrency_safe`, `rule_matcher`, `args_preview`) via the `tool_metadata(...)` helper. Permissions, safety floors, context memory, audit journaling, and output budgets plug in as `HookChain` entries — not loop-body logic. Config is JSON with a `providers[]` list + `router{}` alias table.

## Install

### From PyPI (end users)

```bash
pip install aura-agent           # base install
pip install "aura-agent[openai]" # + OpenAI provider
pip install "aura-agent[all]"    # + all providers (OpenAI, Anthropic, Ollama, web search)
```

Or with uv:

```bash
uv add aura-agent
uv add "aura-agent[all]"
```

After install, the `aura` command is on your PATH.

### From source (contributors)

```bash
git clone https://github.com/LordFoxFairy/Aura.git
cd Aura
uv sync --extra all --extra dev
uv run aura --version
```

## Quickstart

1. Set your API key (pick one):

   ```bash
   export OPENAI_API_KEY="sk-..."
   # or
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

2. Run the REPL:

   ```bash
   uv run aura
   ```

   You'll get `aura>` as the prompt. Type anything to send it to the model. `/help` lists commands.

## Configuration

Aura reads JSON config from (in precedence order):

1. `$AURA_CONFIG` env var
2. `./.aura/config.json` (project)
3. `~/.aura/config.json` (user)
4. Built-in defaults

Example `~/.aura/config.json` with three providers + router aliases:

```json
{
  "providers": [
    {
      "name": "openai",
      "protocol": "openai",
      "api_key_env": "OPENAI_API_KEY"
    },
    {
      "name": "openrouter",
      "protocol": "openai",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY"
    },
    {
      "name": "deepseek",
      "protocol": "openai",
      "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
      "api_key_env": "DEEPSEEK_API_KEY",
      "params": {"temperature": 0.2}
    }
  ],
  "router": {
    "default": "openai:gpt-4o-mini",
    "opus":    "openrouter:anthropic/claude-opus-4",
    "fast":    "openrouter:anthropic/claude-3.5-haiku"
  },
  "tools":   { "enabled": ["read_file", "write_file", "edit_file", "grep", "glob", "bash", "todo_write", "ask_user_question", "web_fetch"] },
  "storage": { "path": "~/.aura/sessions.db" }
}
```

Any OpenAI-compatible endpoint works via `"protocol": "openai"` + `base_url`. `params` are forwarded as kwargs to the LangChain constructor (temperature, max_tokens, timeout, etc.).

Permission rules live separately in `.aura/settings.json` (committed, shared) and `.aura/settings.local.json` (gitignored, per-machine). See the permissions section below.

## Built-in tools

Eleven tools ship with Aura:

| Tool | Purpose | Notes |
|---|---|---|
| `read_file` | UTF-8 read with 1 MB cap | `offset` / `limit` for partial reads; partial reads tracked so `edit_file` won't operate on a slice |
| `write_file` | Create / overwrite | Parent dirs auto-created; must-read-first guard for overwrites |
| `edit_file` | In-place string replace | Must-read-first invariant (never / stale / partial); mixed CRLF+LF preserved per-line; `old_str=""` on non-existent path creates |
| `grep` | Ripgrep-backed content search | Modes: `files_with_matches` (default) / `content` / `count`; `-A`/`-B` context; multiline; `--type`; sentinel separators for paths with `-<digits>-` |
| `glob` | Path search | `mtime`-sort default (newest first); `alphabetical` opt-in |
| `bash` | Shell command | Native async; `AbortSignal`-equivalent cancellation kills child; 30 KB per-stream output cap; 100 MB hard memory ceiling; Tier A safety floor (zsh-dangerous builtins, CR parser-diff, command substitution, cd+git compound) |
| `todo_write` | Session task list | Exactly-one `in_progress` cardinality enforced |
| `ask_user_question` | Ask the user mid-turn | Free-text or 2–6 option list; TTY-gated |
| `web_fetch` | Fetch URL + content | SSRF-hardened (blocks cloud-metadata IPs, private-network, loopback); prompted — not auto-allow |
| `task_create` | Spawn subagent (fire-and-forget) | Returns `task_id` immediately; child runs detached on the event loop via `asyncio.create_task`; child gets a `:memory:` SessionStorage + own Context + no MCP + no skills (MVP); `task_create` + `task_output` stripped from the child's tool list (no recursion in MVP) |
| `task_output` | Fetch subagent transcript | Read-only, concurrency-safe; raises on unknown `task_id` |

All are `BaseTool` subclasses with Aura metadata (`tool_metadata(...)` sets `is_read_only`, `is_destructive`, `is_concurrency_safe`, `rule_matcher`, `args_preview`, `max_result_size_chars`).

## Skills

User-authored skills loaded from `~/.aura/skills/*.md` (global) and `<cwd>/.aura/skills/*.md` (project-local). Each skill registers as a slash-invokable command `/<name>`; the skill's markdown body is injected into the model's context on invocation as a `<skill-invoked name="...">` HumanMessage tag — same tag-based mechanism as project memory and nested memory. The skills listing also surfaces as a `<skills-available>` tag so the model knows what's available even before invocation. User-layer wins on cross-layer name collision; intra-layer duplicates drop with a `skill_parse_failed` journal event.

Example `~/.aura/skills/refactor-python.md`:

```markdown
---
name: refactor-python
description: Python refactoring assistant — extracts functions, renames cleanly.
---
When the user invokes this skill, you are in refactor-python mode. Work in small
diffs. Always run the test suite after each edit. Prefer `rg`-backed grep over
semantic guesses. Never rename across the public API boundary without asking.
```

The REPL shows `/refactor-python` in `/help` automatically. Directory-form skills, file-pattern triggers, LLM-invokable `SkillTool`, and hot reload are deferred to 0.2.x.

## MCP

MCP servers connect via `langchain-mcp-adapters` (stdio transport; HTTP/SSE deferred). Discovered tools are auto-namespaced `mcp__<server>__<tool>` and registered dynamically into the `ToolRegistry` after `Agent.aconnect()`; the model is re-bound with the updated tool set via `AgentLoop._rebind_tools`. MCP prompts surface as slash commands `/<server>__<prompt>` with `source="mcp"`. Aura applies conservative defaults to each MCP tool: `is_destructive=True`, `is_concurrency_safe=False`, `max_result_size_chars=30_000` — because the library can't speak to the semantics of arbitrary server tools.

Config block:

```json
{
  "mcp_servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
      "env": {},
      "transport": "stdio",
      "enabled": true
    }
  ]
}
```

Per-server failures are isolated (the library's bulk `get_tools()` crashes on any single failure; `MCPManager` loops per server). Each failure journals `mcp_connect_failed` and the other servers still come up. Teardown reflects over `aclose` / `close` because the library has no teardown API today.

## Compact

`/compact` (or `Agent.compact(source="manual")`) summarizes the history prefix and rebuilds the conversation around a `<session-summary>` HumanMessage tag. Session state you can't regenerate is preserved across the boundary: `_read_records` (must-read-first fingerprints), `_invoked_skills`, `state.custom['todos']`, session permission rules, cumulative `total_tokens_used`. Discovery caches — project memory, rules, `_nested_fragments`, `_matched_rules` — are cleared and re-discovered on the next turn. `SystemMessage` is untouched (prompt-cache friendly).

Use when a session has accumulated many turns but the model is still doing useful work — compact preserves continuity without paying for the full history on every subsequent request. Last `KEEP_LAST_N_TURNS=3` turns stay raw. Summary is produced by the unbound raw model (no tool calls possible by construction). Auto-trigger at `AUTO_COMPACT_THRESHOLD=150_000`, selective file re-injection, and reactive recompact on "prompt too long" API errors are deferred to 0.4.1.

## Subagents

`task_create(description, prompt)` schedules a subagent via `asyncio.create_task(run_task(...))` and returns a `task_id` immediately; the parent's loop continues without blocking. The subagent is a fresh `Agent` built by `SubagentFactory` with the same config + model + permissions but its own `:memory:` SessionStorage and its own Context. `task_output(task_id)` fetches the current `TaskRecord` (status + final result + error). `/tasks` lists the 20 most recent records. `Agent.close()` cancels any in-flight subagents (parent cancel cascades).

The unified rendering channel (prompt-toolkit Application refactor with foldable regions + dialog queue) is the other half of the original Phase E design and is deferred to 0.6.0. Today subagent output is visible via `task_output` + `/tasks` only — there's no mid-turn streaming into the REPL pane.

## Permissions & safety

Aura has three layers — all composed via `HookChain.pre_tool` in `build_agent`:

1. **`bash_safety` hook** (position 0, **hard floor** — cannot be overridden by rules or `--bypass-permissions`). Rejects shell commands matching four Tier A patterns:
   - `zsh_dangerous_command` (18 builtins like `zmodload`, `syswrite`, `zf_rm`)
   - `cr_outside_double_quote` (shell-quote vs bash parser differential)
   - `command_substitution` (`$(...)` / backticks / `bash -c` / `eval`)
   - `cd_git_compound` (cd into malicious repo + git command = fsmonitor RCE)

2. **Permission hook** (CLI-installed). Consults rules in `.aura/settings.json` + `.aura/settings.local.json`:
   - `mode`: `default` / `accept_edits` / `plan` / `bypass`
   - `allow: ["read_file", "bash(npm *)", "grep"]` — rule strings
   - `safety_exempt: ["~/safe/workspace/**"]` — paths bypassing `DEFAULT_PROTECTED_WRITES` / `DEFAULT_PROTECTED_READS`
   - Prompts the user with a list-select dialog (`Allow once` / `Allow always` / `Deny`) when a rule doesn't match. `Allow always` persists to session OR project settings.

3. **`must_read_first` hook** (tool-intrinsic, position end). Gates `edit_file` and `write_file` overwrites on session-scoped read state (Context `_read_records` map of `(mtime, size, partial)` fingerprints). Rejects with distinct `never_read` / `stale` / `partial` reasons so the LLM can self-correct.

`bash` and destructive tools never auto-allow. `write_file` to paths matching `**/.git/**`, `**/.ssh/**`, `/etc`, shell rc files, etc. is blocked by `DEFAULT_PROTECTED_WRITES` regardless of `allow` rules.

## Slash commands

Slash commands are backed by a unified `CommandRegistry` (since 0.1.1) that also hosts skill commands (one per loaded skill) and MCP prompts (one per remote `<server>__<prompt>`). Run `/help` for the live list.

| Command | Effect |
|---|---|
| `/help` | show command list (live-enumerated, includes skills + MCP prompts) |
| `/exit` | exit the REPL (also Ctrl+D) |
| `/clear` | clear the current session's history and turn counter |
| `/model` | show the current default model + router aliases |
| `/model <alias>` | switch via router alias (e.g. `/model opus`) |
| `/model <provider:model>` | switch directly (e.g. `/model openrouter:anthropic/claude-opus-4`) |
| `/compact` | summarize history + preserve session state (see Compact section) |
| `/tasks` | list recent subagent tasks (id prefix + status + description, top 20) |
| `/<skill-name>` | invoke a user-authored skill (auto-registered from `~/.aura/skills/` + `<cwd>/.aura/skills/`) |
| `/<server>__<prompt>` | invoke an MCP prompt (auto-registered when MCP server connects) |

## Default hooks

`build_agent(config)` wires these automatically:

- **`bash_safety`** (`pre_tool` at position 0) — hard-floor Tier A safety
- **Permission hook** (`pre_tool`, CLI-installed) — rule matching + user prompt
- **`must_read_first`** (`pre_tool`, end) — read-before-modify guard
- **Token accounting** (`post_model`) — `state.total_tokens_used` per session
- **Output size budget** (`post_tool`, default 50 000 chars) — caps large tool output with optional spill-to-disk

The loop enforces `max_turns=50` directly (not a hook — see `aura/core/loop.py`), matching claude-code's `query.ts:1705` pattern. Natural stops yield `Final(reason="natural")`; cap hits yield `Final(reason="max_turns")`.

## Extension points

### Add a custom tool (production pattern)

Subclass `BaseTool` directly and attach Aura metadata:

```python
from typing import Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from aura.schemas.tool import tool_metadata
from aura.core.permissions.matchers import exact_match_on


class SearchParams(BaseModel):
    query: str = Field(description="search query")


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = "Search the web and return top hits."
    args_schema: type[BaseModel] = SearchParams
    metadata: dict[str, Any] | None = tool_metadata(
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_size_chars=30_000,
        rule_matcher=exact_match_on("query"),
    )

    async def _arun(self, query: str) -> dict[str, Any]:
        # ... your implementation ...
        return {"hits": [...]}


web_search: BaseTool = WebSearch()
```

Inject it at agent construction:

```python
from aura import build_agent, load_config
agent = build_agent(load_config(), available_tools={"web_search": web_search})
```

> `build_tool(...)` in `aura.tools.base` is a **test-only** factory — for production tools, always subclass `BaseTool` as above.

### Add a custom hook (e.g. audit log)

```python
from typing import Any
from langchain_core.tools import BaseTool

from aura.core.hooks import HookChain
from aura.schemas.tool import ToolResult
from aura.schemas.state import LoopState


async def audit(*, tool: BaseTool, args: dict[str, Any], result: ToolResult, state: LoopState, **_) -> ToolResult:
    print(f"[audit] {tool.name} -> ok={result.ok}")
    return result


hooks = HookChain(post_tool=[audit])
agent = build_agent(load_config(), hooks=hooks)
```

## Design docs

- Spec: `docs/specs/2026-04-17-aura-mvp-design.md`
- Plan: `docs/plans/2026-04-17-aura-mvp.md`
- Phase E design: `docs/specs/2026-04-21-phase-e-subagent.md`
- Research — claude-code design principles: `docs/research/claude-code-design-principles.md`
- Manual: `docs/manual/` (per-module how-tos)
- Changelog: `CHANGELOG.md` (v0.1.0 → v0.7.4)

## Status

**0.7.4 — eleven incremental releases since the walking-skeleton MVP.** Own-loop, JSON config, 11 built-in tools, hook-based extensibility, end-to-end permissions + safety (4 modes), skills, MCP, compact + auto-compact, fire-and-forget subagents with full inheritance, live status bar with context-pressure indicator, inline permission prompts. LangChain client-layer only. 1107 tests green.

| Tag | Theme |
|---|---|
| `v0.1.0` | Walking skeleton — own-loop, 9 tools, permissions + safety, 3-layer context memory |
| `v0.1.1` | Unified `Command` abstraction — `CommandRegistry` + `Command` Protocol (prereq for Skills + MCP) |
| `v0.2.0` | Skills MVP — `~/.aura/skills/*.md` + `<cwd>/.aura/skills/*.md` with slash-command invocation |
| `v0.3.0` | MCP via `langchain-mcp-adapters` — stdio transport, `mcp__<server>__<tool>` namespace, dynamic tool registry |
| `v0.4.0` | Compact — `Agent.compact()` summarizes history; `_read_records` / skills / todos / session rules survive |
| `v0.5.0` | Task tool — fire-and-forget subagent via `asyncio.create_task`; `task_create` / `task_output` / `/tasks` |
| `v0.5.1` | Auto-compact trigger, session-scoped journal, claude-code design study |
| `v0.6.0` | Close parity gaps — subagent skill/MCP inheritance, permission inheritance, full transcript accumulation |
| `v0.7.0` | REPL + render polish — tool-error hints, SlashCommandCompleter, welcome banner |
| `v0.7.1` | Context-aware bottom bar — live token display, cached/pinned split, color-coded pressure |
| `v0.7.2` | Mode plumbing to status bar, `AuraConfig.context_window` override, 512k unknown-model default |
| `v0.7.3` | Monochrome bottom bar, noreverse footer, compact 3-line welcome, post-turn status checkpoint |
| `v0.7.4` | Inline permission prompts (no more bordered dialogs), web_search DuckDuckGo-only, leftover closure |

**Deferred to future phases:**
- Full-screen prompt-toolkit `Application` mode — persistent status bar that doesn't disappear during response streaming. Today's post-turn checkpoint is the pragmatic middle ground.
- Tauri desktop wrapper — not scheduled.
- Additional web-search backends (Tavily / Serper) — removed in v0.7.4. DuckDuckGo is zero-config and covers the general-purpose search need.

## License

MIT.
