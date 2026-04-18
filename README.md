# Aura

A lightweight Python agent with an explicit async loop, pluggable LLM providers, and a hook-based extension model. Built on LangChain as a client-only layer — no LangGraph, no LangChain agents, no LCEL.

## Design in one paragraph

Aura owns the agent loop. LangChain is used strictly as a uniform `BaseChatModel` client over OpenAI / Anthropic / Ollama and any OpenAI-compatible endpoint (OpenRouter, DeepSeek-OAI mode, self-hosted). The loop dispatches tool calls through a registry of `AuraTool` instances; permission, size budgets, token accounting, and auditing plug in as `HookChain` entries, not as loop-body logic. Config is JSON with a `providers[]` list + `router{}` alias table.

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
    "fast":    "openrouter:anthropic/claude-3.5-haiku",
    "glm":     "deepseek:glm-5"
  },
  "tools":   { "enabled": ["read_file", "write_file", "bash"] },
  "storage": { "path": "~/.aura/sessions.db" }
}
```

Any OpenAI-compatible endpoint works via `"protocol": "openai"` + `base_url`. `params` are forwarded as keyword arguments to the LangChain constructor (temperature, max_tokens, timeout, etc.).

## Slash commands

| Command | Effect |
|---|---|
| `/help` | show command list |
| `/exit` | exit the REPL (also Ctrl+D) |
| `/clear` | clear the current session's history and turn counter |
| `/model` | show the current default model + router aliases |
| `/model <alias>` | switch via router alias (e.g. `/model opus`) |
| `/model <provider:model>` | switch directly (e.g. `/model openrouter:anthropic/claude-opus-4`) |

## Built-in safety

By default, `build_agent(config)` wires three hooks:

- **Max turns** (`20`) — prevents runaway loops from draining your wallet
- **Token accounting** — tracks `total_tokens_used` per conversation
- **Output size budget** (`50,000` chars) — caps large tool outputs with optional spill-to-disk

Destructive tools (`write_file`, `bash`) prompt with `y / N / a` before running. `a` adds the tool to a session-scope allowlist.

## Extension points

Add a custom tool:

```python
from aura.tools import build_tool, AuraTool, ToolResult
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str

async def _search(params):
    ...
    return ToolResult(ok=True, output={"hits": [...]})

search: AuraTool = build_tool(
    name="web_search",
    description="search the web",
    input_model=SearchParams,
    call=_search,
    is_read_only=True,
    max_result_size_chars=30_000,
)
```

Inject it:

```python
from aura import build_agent, load_config
agent = build_agent(load_config(), available_tools={"web_search": search})
```

Add a custom hook (e.g. audit log):

```python
from aura.core import HookChain, PostToolHook
from aura.tools import AuraTool, ToolResult

async def audit(*, tool: AuraTool, params, result: ToolResult, state, **_) -> ToolResult:
    print(f"[audit] {tool.name} -> ok={result.ok}")
    return result

hooks = HookChain(post_tool=[audit])
agent = build_agent(load_config(), hooks=hooks)
```

## Design docs

- Spec: `docs/specs/2026-04-17-aura-mvp-design.md`
- Plan: `docs/plans/2026-04-17-aura-mvp.md`
- Manual: `docs/manual/` (per-module how-tos)

## Status

MVP (phase 1). Walking skeleton with own-loop, JSON config, 3 built-in tools, hook-based extensibility. LangChain client-layer only. `260+` tests green; `aura --version` reports `0.1.0`.

Deferred to later phases: MCP integration, skills, multi-session / resume, Tauri desktop, subagents, auto-compact.

## License

MIT.
