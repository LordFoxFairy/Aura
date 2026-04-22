# Aura

A lightweight Python agent with an explicit async loop, pluggable LLM providers, and a hook-based extension model. Built on LangChain as a client-only layer — no LangGraph, no LangChain agents, no LCEL.

## Design in one paragraph

Aura owns the agent loop. LangChain is used strictly as a uniform `BaseChatModel` client over OpenAI / Anthropic / Ollama and any OpenAI-compatible endpoint (OpenRouter, DeepSeek, self-hosted). The loop dispatches tool calls through a `ToolRegistry` of LangChain `BaseTool` subclasses that carry Aura-specific metadata (`is_read_only`, `is_destructive`, `is_concurrency_safe`, `rule_matcher`, `args_preview`) via the `tool_metadata(...)` helper. Permissions, safety floors, context memory, audit journaling, and output budgets plug in as `HookChain` entries — not loop-body logic. Config is JSON with a `providers[]` list + `router{}` alias table.

## Install

```bash
# clone, then
uv sync --extra openai       # or --extra anthropic / --extra ollama / --extra all
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

Nine tools ship with Aura:

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

All are `BaseTool` subclasses with Aura metadata (`tool_metadata(...)` sets `is_read_only`, `is_destructive`, `is_concurrency_safe`, `rule_matcher`, `args_preview`, `max_result_size_chars`).

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

| Command | Effect |
|---|---|
| `/help` | show command list |
| `/exit` | exit the REPL (also Ctrl+D) |
| `/clear` | clear the current session's history and turn counter |
| `/model` | show the current default model + router aliases |
| `/model <alias>` | switch via router alias (e.g. `/model opus`) |
| `/model <provider:model>` | switch directly (e.g. `/model openrouter:anthropic/claude-opus-4`) |

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
- Phase E design (deferred): `docs/specs/2026-04-21-phase-e-subagent.md`
- Research — claude-code design principles: `docs/research/claude-code-design-principles.md`
- Manual: `docs/manual/` (per-module how-tos)

## Status

**MVP (phase 1, walking skeleton).** Own-loop, JSON config, 9 built-in tools, hook-based extensibility, end-to-end permissions + safety. LangChain client-layer only. 827 tests green; `aura --version` reports `0.1.0`.

**Deferred to later phases:**
- Subagent (`Task` tool) + unified prompt-toolkit rendering channel — see `docs/specs/2026-04-21-phase-e-subagent.md`
- MCP integration, skills, Tauri desktop, auto-compact
- Token-budget diminishing-returns detection (advisory; `max_turns=50` is the hard floor today)

## License

MIT.
