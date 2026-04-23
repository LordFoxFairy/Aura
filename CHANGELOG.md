# Changelog

Notable changes to Aura. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.8.3] — Markdown rendering + retry logic + git slash commands

Three more subagents in parallel. 1300 tests green.

### Shipped

- **Rich markdown rendering for assistant output** — Renderer buffers `AssistantDelta` text, flushes on any non-delta event or `finish()`, and routes text with markdown markers through `rich.markdown.Markdown(code_theme="monokai", inline_code_lexer="python")`. Plain text bypasses rich overhead. `UIConfig.markdown: bool = True` toggle. Renderer gains a `markdown: bool = True` kwarg (backward-compat default).
- **Retry logic with exponential backoff** — new `aura/core/retry.py` with `with_retry(fn, max_attempts, base_delay_s, max_delay_s, jitter, retriable)`. Classifies transient errors by exception class name OR message substring (no SDK imports, durable across provider versions). `_RETRIABLE_CLASS_NAMES`: RateLimitError / ServiceUnavailableError / APITimeoutError / APIConnectionError / InternalServerError / ConflictError. Substring: rate limit / 503 / 502 / 504 / timeout / overloaded / try again. Deny-list wins over allow-list (auth errors containing "timeout" don't retry forever). `asyncio.CancelledError` always propagates. Wrapped ONLY around the narrow `model.ainvoke` site. `AuraConfig.retry: RetryConfig | None`.
- **Git-aware slash commands** — `/status`, `/diff`, `/log`.
  - `/status` — `git status --short --branch`, formatted with branch bold cyan, file status codes color-coded (M/A/D/??), empty tree → "working tree clean".
  - `/diff` — stat summary by default; `/diff --full` for full patch; `/diff --staged` for index. Direct ANSI passthrough from git, truncated at 500 lines.
  - `/log` — `git log --oneline --decorate --color=always -20` default; `/log N` for last N commits (clamped [1, 100]).
  - All three: graceful "not a git repository" / "git not installed" / timeout errors; never crash the REPL.
  - Design note: commands use `writer: Callable[[str], object]` instead of `Console` so `aura/core/` keeps its "no rich imports" invariant (enforced by `test_core_does_not_import_ui_frameworks`).

### Changed

- `aura/cli/render.py` — buffer-and-flush path, markdown routing, heuristic `_looks_like_markdown()`.
- `aura/config/schema.py` — `UIConfig.markdown`, `RetryConfig` + `AuraConfig.retry`.
- `aura/core/loop.py` — `retry_config` ctor kwarg; `with_retry` around `_invoke_model`.
- `aura/core/agent.py` — `_build_loop` passes `retry_config=self._config.retry`.
- `aura/cli/commands.py` — registers `/status`, `/diff`, `/log`.
- `aura/core/commands/git_commands.py` (NEW) — 3 git commands + `_git()` shell-out helper.
- `aura/core/retry.py` (NEW) — retry helper + classifier.

### Stats

- 1300 tests pass (1248 + 52 new: 12 markdown + 23 retry + 17 git). Lint + mypy clean.

## [0.8.2] — MCP HTTP/SSE + /export + ASCII welcome

Three more subagents in parallel. Zero merge conflicts, 1248 tests green.

### Shipped

- **MCP HTTP/SSE transport** — `MCPServerConfig.transport` widens from `Literal["stdio"]` to `Literal["stdio", "sse", "streamable_http"]` (names verified against `langchain_mcp_adapters.sessions`). New `url: str | None` and `headers: dict[str, str]` fields. Pydantic validator enforces `stdio↔command` vs `sse|http↔url` mutual exclusivity. `MCPManager._build_connections` dispatches per transport; unsupported library versions raise a clear `AuraConfigError` at config-load (not runtime). Disabled servers skip the transport gate (backward compat).
- **`/export` slash command** — saves the session transcript to a file. `/export` (no args) writes markdown to `~/.aura/exports/aura-session-<timestamp>.md`; `/export <path>` with extension-inferred format; `/export --format json` for structured output. Markdown includes envelope (session_id / timestamp / model / cwd / turns / total_tokens), `## Turn N (role)` sections, tool-call lists, fenced tool-result blocks with language hints. JSON preserves `role` / `content` / `tool_calls` / `tool_call_id`. Bad paths return error CommandResult (REPL never crashes).
- **ASCII welcome banner** — panel-framed `✱ AURA` wordmark + version in cyan, followed by 3 dim info lines (`model:`, `cwd:`, `tip:`), footer with quit hints. Rotating tip pulled randomly from a curated list of 10 (all verified against current features — `/model`, `/export`, `/compact`, `/tasks`, Tab-to-amend, ctrl+e explain, shift+tab mode cycle, etc.). Renders cleanly in 80-col terminals.

### Changed

- `aura/config/schema.py:MCPServerConfig` — transport Literal widened, url/headers fields, validator.
- `aura/core/mcp/manager.py` — `_supported_transports()` probe, per-transport connection building.
- `aura/core/commands/export.py` (NEW) — ExportCommand implementation.
- `aura/cli/commands.py` — registers /export.
- `aura/cli/repl.py` — `_STARTUP_TIPS` tuple, banner rewrite with Panel + wordmark + rotating tip.
- Tests: `tests/test_mcp_config.py` (17 new), `tests/test_export_command.py` (13 new), `tests/test_repl.py` (welcome banner reassertions + version-drift test).

### Stats

- 1248 tests pass (1217 + 31 new). Lint + mypy clean.

## [0.8.1] — Background bash + Plan mode tools + /model runtime switch

3 parallel subagents again, merged zero conflicts. Each one closes a user-facing gap against claude-code.

### Shipped

- **`bash_background`** — fire-and-forget shell task. Returns a `task_id` immediately; process runs detached, stdout/stderr streamed line-by-line into `TaskRecord.progress.recent_activities` (ring-buffer cap 20, `[out]`/`[err]` prefixes). Stall detector fires a single `[stalled?]` marker after 30s of silence with output already captured (re-arms on new output). Shutdown ladder SIGTERM → 3s → SIGKILL on `task_stop`. Same bash-safety floor as the blocking `bash` tool (no `$(...)` / backticks / `-c`). `TaskRecord.kind: Literal["subagent", "shell"]` + `task_list` filter distinguishes shell tasks from subagent tasks.
- **`enter_plan_mode` / `exit_plan_mode`** — LLM can programmatically flip mode. `enter_plan_mode(plan: str)` takes a markdown plan and sets mode → `plan`; `exit_plan_mode(to_mode="default")` restores. Both are **mode-exempt** so the LLM can always exit. Plan-mode enforcement tightened in `aura/core/hooks/permission.py`: allow list `{read_file, grep, glob, web_fetch, web_search, task_get, task_list}` + exempt `{enter_plan_mode, exit_plan_mode}`; everything else (including unknown tools) fails closed with `permission_decision` reason=`plan_mode_blocked`. Safety hooks still run before the mode branch.
- **`/model` slash command** — runtime router switching. `/model` (no args) prints the current spec + aliases; `/model <alias>` / `/model provider:model` calls `Agent.switch_model(spec)` which re-resolves via `llm.resolve` + `llm.create`, replaces `self._model`, rebuilds the loop. History survives. `config.router["default"]` is untouched (static config ≠ live spec). Missing credentials / unknown provider raise `AuraConfigError` → printable REPL error, not a crash.

### Changed

- `aura/core/tasks/types.py` — `TaskKind` Literal, `kind` / `metadata` fields, `TaskProgress.line_count`.
- `aura/core/tasks/store.py` — `create(kind=..., metadata=...)`, `list(kind=...)`, `record_shell_line`, `record_shell_marker`.
- `aura/tools/task_list.py`, `task_get.py`, `task_stop.py` — kind filter / surface / bimodal stop (Task.cancel for subagents; Process.terminate→kill for shells).
- `aura/core/agent.py` — `_running_shells` map, 3 new stateful tools wired, `switch_model` uses live spec field, `close()` kills orphan shells.
- `aura/core/hooks/permission.py` — plan-mode read-allow + tool-exempt lists.
- `aura/core/commands/builtin.py` — `/model` command with new format + `AuraConfigError` handling.
- `aura/config/schema.py` — default `tools.enabled` gains `bash_background`, `enter_plan_mode`, `exit_plan_mode`.

### Stats

- 1217 tests pass (1181 + 36 new). Lint + mypy clean. Three subagents, zero merge conflicts.

## [0.8.0] — Task lifecycle tools + StatusLine hook + ctrl+e explain

Major: Task subsystem gets a real lifecycle API (not just fire-and-forget), the bottom bar becomes user-extensible via a shell hook, and the permission widget gains ctrl+e for an inline explanation block. Shipped via 3 parallel subagents, merged zero conflicts.

### Shipped

- **Task lifecycle tools** — `task_get`, `task_list`, `task_stop` join `task_create` + `task_output`. Maps almost 1:1 onto claude-code's `TaskGetTool` / `TaskStopTool` / `TaskListTool`.
  - `task_get` — returns full `TaskRecord` (status, started_at, finished_at, duration, final_result, error, progress) with an `include_messages` flag for the full transcript.
  - `task_stop` — cancels a running subagent: looks up the `asyncio.Task` handle on `Agent._running_tasks`, cancels, awaits unwind with timeout, sets `status=cancelled`.
  - `task_list` — paginated + status-filterable listing + a counts summary (`{running, completed, failed, cancelled}`).
  - **Progress tracking** — `TaskRecord.progress: TaskProgress` (tool_count / token_count / last_activity_at / recent_activities ring-buffer cap 5). Updates from `run_task` as child tool events arrive.
  - Slash commands: `/task-get <id>`, `/task-stop <id>` (short-id prefix resolution).
  - Subagent inheritance: `task_get` / `task_list` / `task_stop` flow to children (they operate on the child's own store — safe no-op); `task_create` / `task_output` stay stripped (no recursion).

- **Configurable StatusLine hook** — `.aura/settings.json` gains a new `statusline` block:
  ```json
  { "statusline": {"command": "bash -c ~/bin/aura-statusline.sh", "timeout_ms": 500, "enabled": true} }
  ```
  When set, each bottom-toolbar render executes the command with a stable v1 JSON envelope (`model`, `context_window`, `tokens`, `mode`, `cwd`, `last_turn_seconds`) on stdin. ANSI-escaped stdout becomes the bar. Timeouts / non-zero exits / crashes silently fall back to the default rendering — the bar never degrades to garbage. Matches claude-code's `StatusLine.tsx` hook contract.
  - pt 3.0.52's `bottom_toolbar` must return synchronously; implemented as cached-value + async-refresh with `app.invalidate()` so each render shows the latest hook output without blocking.

- **ctrl+e to explain** on the permission widget — toggles an inline `┌ Explanation … └` block with *What this tool does* / *Arguments* (pretty-printed 2-column, truncated at 80 chars) / *Risk* (one-liner per tag) / *What happens if you approve* (per-tool verb). **Static, no LLM call** — matches claude-code's UX surface without the latency / cost / race conditions of a separate inference. Ignored while in Tab-amend feedback mode so typing doesn't steal keystrokes.

### Changed

- `aura/core/tasks/types.py` — `TaskRecord.progress` field; `TaskProgress` dataclass.
- `aura/core/tasks/store.py` — `record_activity()`, `list(limit=…)`.
- `aura/core/tasks/run.py` — emits progress on each `ToolCallStarted`.
- `aura/core/agent.py` — wires 3 new stateful tools.
- `aura/config/schema.py` — default `tools.enabled` gains `task_get` / `task_list` / `task_stop`.
- `aura/schemas/permissions.py` — new `StatusLineConfig`; `PermissionsConfig.statusline: StatusLineConfig | None`.
- `aura/cli/permission.py` — `_build_explanation`, `c-e` keybinding, widget state `explain_visible`.
- `aura/cli/statusline_hook.py` (NEW) — `run_statusline_command`, `build_envelope`, `STATUSLINE_ENVELOPE_VERSION`.
- `aura/cli/status_bar.py` — `render_bottom_toolbar_with_hook`.
- `aura/cli/repl.py` — cached-hook-output + async-refresh pattern.

### Stats

- 1181 tests pass (baseline 1125 + 56 new). Lint + mypy clean. Three subagents produced independent diffs with strict file ownership; integration merge required no manual fixups.

## [0.7.6] — 1:1 claude-code parity polish

Owner asked for 1:1 polish against claude-code's real source. Audited three surfaces in parallel (permission widget hotkeys, main prompt footer, per-tool result rendering), ranked gaps by value × cost, shipped the top 4. Skipped the rest with documented reasoning.

### Shipped

- **Error hints propagate to the LLM context.** Aura's error-pattern-to-hint mapping (ripgrep, "has not been read yet", permission-denied, etc.) was a user-facing-only panel; the model never saw it, so recovery in the next turn was worse than it could be. Moved the hint table into `aura/tools/errors.py` (shared between UI and the loop's ToolMessage serializer). Loop now appends `Hint: {text}` to the error string the model sees. The user's red panel keeps its prominent UX weight.
- **Bash stdout/stderr streaming.** `aura/tools/bash.py` already captured output in ring buffers; it just didn't emit progress events. New `ToolCallProgress` event type, context-var-based progress callback plumbing (`aura/tools/progress.py`), renderer prints each chunk dim with a `│ ` prefix. Both stdout and stderr flow independently; spinner stops on first progress event.
- **Tab to amend on permission widget.** Mirrors claude-code's `useShellPermissionFeedback.ts:51-80`. In the widget, Tab toggles a feedback-input buffer under the options — user types free-text, Enter commits the selected option with the feedback, Esc aborts feedback and returns to option mode. `AskerResponse` gains a `feedback: str = ""` field; the "user denied" message sent to the LLM includes `— note: {feedback}` when non-empty; journal event carries it as an optional key. Footer text cycles: `Tab to amend` on options, `Add feedback (Enter to submit, Esc to cancel)` in feedback mode.
- **Shift+Tab cycles permission mode.** Mirrors claude-code's `PromptInputFooterLeftSide.tsx:360-368`. At the main prompt, Shift+Tab cycles `default → accept_edits → plan → default`. `bypass` is explicitly skipped (dangerous; only set via `--bypass-permissions` CLI flag). New `Agent.set_mode(mode)` with validation. Esc at the prompt resets to `default` (claude-code convention). Confirmation prints dim to the scrollback so the transcript records the change; bottom toolbar re-reads `agent.mode` on the next render.

### Skipped (deliberate)

- **ctrl+e to explain** (`PermissionExplanation.tsx:92-147`) — requires a separate LLM request per invocation. Added latency + cost + request volume. User can ask the model "explain this" inline instead; no UI-layer primitive needed.
- **Custom StatusLine hook** (`StatusLine.tsx:30-34`) — user-level customization (shell command producing the status text). Nice-to-have, not parity-blocking. Defer until real demand.
- **Team / PR / Voice / VIM / Remote** footer segments — Anthropic-specific features. Aura has no analog; porting would require the underlying features first.

### Kept (Aura's existing wins, don't "align" away)

- Error panel with red border + multi-line body for tool failures (more scannable than claude-code's inline text).
- Per-tool hint mapping (teaches the model recovery without coaching).
- Graceful degradation on malformed tool output.
- Post-turn status checkpoint (claude-code hides its status during streaming with no mitigation).

### Added files

- `aura/tools/errors.py` — shared hint table + `hint_for_error()` function.
- `aura/tools/progress.py` — ContextVar-based progress callback plumbing.

### Stats

- 1125 tests (baseline 1094, +31 new). Lint + mypy clean. Three parallel subagent diffs merged zero conflicts. Dogfooded: 3 widget scenarios (Tab-commit-with-feedback, Tab-Esc-abort, arrow-nav-and-Enter) all pass in pty.

## [0.7.5] — Interactive arrow-key permission widget + FIFO prompt mutex

User dogfooded v0.7.4 and flagged two real problems: the prompt was typed-number-to-select (↑/↓ leaked through as raw escape codes) and the thinking spinner kept animating behind the prompt. Also pointed at claude-code's actual permission widget (screenshot) asking for that exact layout.

### Headlines

- `aura/cli/permission.py` now uses an inline `prompt_toolkit.Application` for true arrow-key navigation. ↑/↓ wrap-cursor, Enter / c-m / c-j commit, 1/2/3 number shortcuts jump-and-commit, Esc / Ctrl+C cancel. Rendered in-scrollback with `erase_when_done=True`.
- Layout matches claude-code's permission dialog: horizontal rule separator, bold section title ("Bash command"), dim command preview, verb description, "Do you want to proceed?" heading, options with cyan cursor highlight, footer hints. No box, no glyph+tag noise.
- Option wording: "Yes, and don't ask again for \`<rule>\` in this project" (matches claude-code; was "always allow").
- `aura/cli/user_question.py` gets the same arrow-key widget for multi-choice; free-text path prints question inline + reads via PromptSession.

### Added

- `aura/cli/_coordination.py` — module-level single-writer state for two cross-module concerns: (a) spinner pause callback (REPL registers `ThinkingSpinner.stop` for the turn's lifetime, permission asker calls it before pt takes the screen — fixes the "✳ Concocting… ❯ 221" garble); (b) `prompt_mutex()` returns a process-wide `asyncio.Lock` that every interactive widget acquires, so concurrent prompts from parent + subagent serialize FIFO (mirrors claude-code's `promptQueue` at `REPL.tsx:1110`).
- Turn-duration piece on both status-bar surfaces (appended as the last segment, elided when 0). Char-count / 4 tokenization via `_format_duration`; under-60s shows one decimal, ≥60s shows integer seconds.
- Compact-warning piece (`⚠ compact soon`) on both surfaces when input-tokens / context-window ≥ 80%. Constant `_COMPACT_WARN_PCT = 80` in `status_bar.py`.

### Removed

- `_parse_choice` typed-number parser (replaced by key bindings).
- `_build_dialog_text` radiolist body builder.
- `tests/test_permission_dialog_polish.py` (replaced by the widget-level tests).

### Dogfood

Verified via pty-driven isolated tests:
- 4 permission-widget scenarios pass (↓ wrap + Enter → 1; "2" shortcut → 2; bare Enter → default; Ctrl+C → None).
- 2-task concurrent test: `task_two` waits on the mutex until `task_one` commits, then renders; FIFO preserved.
- Full aura-in-pty dogfood: real bash tool-call triggers widget, UP arrow moves cursor, "3" commits No, audit-line `⚠ bash(pwd) — no` prints, tool correctly denied. Spinner stops cleanly before widget takes the screen.

### Stats

- 1094 tests. Lint + mypy clean.

## [0.7.4] — Inline permission prompts + leftover closure

Closing round: owner wanted "不要遗留" — everything half-finished audited and either shipped or removed. Nothing new, just no more stubs.

### Headlines

- Permission prompts and `ask_user_question` prompts both switched from `prompt_toolkit`'s bordered `radiolist_dialog` / `input_dialog` to inline numbered blocks rendered with rich. Matches claude-code's in-scrollback UX; owner: "我不喜欢这种形式的，还是喜欢 CLI 这种交互，claudecode 的".
- `web_search` collapsed to DuckDuckGo-only. `tavily` and `serper` were `Literal` slots that raised `ToolError("not yet implemented")` — removed from the schema entirely. Audit flagged them as dead code; they are.
- `edit_file` had a stub assignment `original_newline = "\n"  # unused on this branch; stub for type-check` — removed.

### Changed

- `aura/cli/permission.py` — full rewrite: no `radiolist_dialog`, no `HTML` formatted text, no `_build_dialog_text`. Renders a rich block (tag glyph + tool + preview + 3 numbered options + "Enter = default · Ctrl+C to cancel" hint), reads the answer via a transient `PromptSession.prompt_async`. Accepts `1`/`y`/`yes`, `2`/`a`/`always`, `3`/`n`/`no`, bare Enter (default: 1 for safe tools, 3 for destructive). Invalid tokens reprompt rather than silently defaulting — safety invariant.
- `aura/cli/user_question.py` — same treatment for the `ask_user_question` tool's CLI asker.
- `aura/config/schema.py` — `WebSearchConfig.provider` narrowed to `Literal["duckduckgo"]`.
- `aura/tools/web_search.py` — dispatch collapsed; module docstring trimmed.
- `aura/tools/edit_file.py` — unused `original_newline` stub deleted.

### Removed

- `tests/test_permission_dialog_polish.py` — every test targeted `_build_dialog_text`, which no longer exists.
- `test_tavily_backend_returns_not_implemented`, `test_serper_backend_returns_not_implemented` — behaviors that no longer exist.

### Stats

- 1107 tests pass (down from 1109 as 2 "not implemented" assertions were removed — the features they asserted on are gone). Lint + mypy clean.

## [0.7.3] — Dogfooded UX polish

Owner ran v0.7.2 and found three real UX problems that tests didn't catch. Fixed all three plus a post-turn checkpoint for when pt's bottom toolbar is hidden during streaming.

### Changed

- **Bottom bar monochrome.** Stripped `_pct_color_tag`; red/yellow/green context-pressure gradient felt noisy against dim surroundings. Uniform `ansigray`.
- **Bottom bar no-reverse.** `Style.from_dict({"bottom-toolbar": "noreverse"})` — pt's default `reverse` produced a two-tone inverted bar on most terminals. Now blends with terminal bg.
- **Welcome banner: 7 → 3 lines.** Branding + `/help` accent + Ctrl+D hint on one line; `model:` and `cwd:` each on their own. Tab / Ctrl+R hints dropped (learned from `/help`).
- **Plaintext api_key warning gated behind `--verbose`.** Was spamming every startup; operators tuned it out. Journal still fires unconditionally so the audit trail is intact.

### Added

- **`Agent.pinned_tokens_estimate`.** Char-count / 4 estimate over the pinned prefix (system msg + memory + rules + skill catalogue + tool schemas). Status bar shows `~Xk pinned` when the provider's `cache_read_input_tokens` is 0 (deepseek, ollama, etc.); real `+Xk cached` when available. No more zero-pinned on first paint.
- **Post-turn status checkpoint.** `_print_post_turn_status` prints a dim `render_status_bar` line after every turn. pt's `bottom_toolbar` only renders while pt owns the screen — during streaming it disappears. This checkpoint keeps the last-turn stats visible mid-conversation.

### Stats

- 1077 tests. Lint + mypy clean.

## [0.7.2] — CLI mode plumbing + 512k default window

- `Agent.__init__` takes a `mode` kwarg; CLI passes the resolved permission mode (config + `--bypass-permissions`) through `build_agent(..., mode=mode)`. Bottom bar reads `agent.mode` instead of a hardcoded `"default"`.
- New `AuraConfig.context_window: int | None` override. `Agent.context_window` honors it; otherwise falls back to `llm.get_context_window`. Useful for 1M extended-context deployments and frontier models not yet in the static table.
- Unknown-model fallback bumped 128k → 512k. Frontier models increasingly ship 400k-1M windows; 128k under-reported and made the %-bar alarmist.
- `test_package.py` switched from literal version pin to SemVer regex — bumps don't cost a test edit.

### Stats

- 1076 tests. Lint + mypy clean.

## [0.7.1] — Context-aware bottom bar + refined token display

- `render_bottom_toolbar_html` — new HTML renderer for pt's bottom bar. Color-coded context-pressure block bar + token counts + pinned/cached separator + mode + cwd. Callable closed over Agent so numbers track live state without polling.
- Split cumulative `total_tokens_used` into `last_input_tokens` (dynamic per-turn prompt size) + `last_cache_read_tokens` (pinned prefix hit rate). Operators saw the old running total climb monotonically regardless of what this turn actually cost — misleading.
- Bottom toolbar callable moved from a no-op footer to a live status surface.

### Stats

- ≈1000 tests.

## [0.7.0] — REPL + rendering polish aligned with claude-code

Multi-subagent parallel polish across REPL, render, and status surfaces. Two subagents stalled on watchdog; main thread completed the missing pieces (`aura/cli/status_bar.py` scaffold, `_render_tool_error` / `_hint_for_error` helpers in `render.py`).

### Added

- Tool-error display: highlighted `●` marker + one-line hint (`_hint_for_error`) mapped from common error substrings (not found, permission denied, ripgrep missing, etc.). Hints ordered most-specific-first so "ripgrep" isn't preempted by "not found".
- Keybinding hints on the welcome banner (moved from an ugly prompt-line footer).

### Changed

- `SlashCommandCompleter` takes a **live** registry getter so skills / MCP commands registered after `PromptSession` construction still tab-complete.
- `complete_while_typing=False` — menu only on Tab, never while typing. The OLD behavior was noisy.
- Welcome banner: `rich.Panel` with cyan border, dim inline keybinding line.

## [0.6.0] — Close the remaining claude-code parity gaps

- **Per-subagent permission inheritance**: subagents inherit the parent's SessionRuleSet and SafetyPolicy.
- **Full transcript accumulation**: subagents record every event (not just `Final`) so `task_output` can replay the run.
- **Skill / MCP sharing with parent**: subagents now get the parent's `SkillRegistry` and `mcp_servers` list — they were empty in v0.5.0. Deleted the v0.5.1 test pinning the (wrong) old contract.
- **Recursion guard only strips `task_create` + `task_output`** from the child's tool list. Child can still use every other tool + skill + MCP service the parent has.

## [0.5.1] — Audit fixes, auto-compact, session-scoped journal, docs

- **Auto-compact triggers**: `Agent.astream` checks `total_tokens_used` against `auto_compact_threshold` after each turn; if over, invokes `compact(source="auto")` inline.
- **Session-scoped journal**: `journal.write` routes to a per-session JSONL via `contextvars` when `session_log_dir` is configured. Two concurrent Agents (subagents, server workers) in the same process get fully separate audit trails.
- **claude-code design principles study** (`docs/research/claude-code-design-principles.md`) — 282 lines of dense notes with file:line citations across the five subsystems (tool, hook/permission, query loop, context/memory, rendering) that shaped Aura's design.

## [0.5.0] — Task tool (scoped Phase E)

Fire-and-forget subagent dispatch. Parent returns immediately with a `task_id`; child runs detached on the event loop; transcripts fetched later via `task_output`. Matches claude-code's `shouldDefer: true` pattern (`TaskCreateTool.ts:67`).

### Headlines

- `task_create` + `task_output` built-in tools; `/tasks` slash command listing recent subagent runs.
- SubagentFactory inherits parent config + model + permissions; subagent OWNS an in-memory `:memory:` SessionStorage, OWNS its own Context, and OWNS its own `HookChain`.
- Parent `Agent._running_tasks: dict[task_id, asyncio.Task]` tracks handles; `Agent.close()` cancels each — parent cancel cascades to subagent.
- Unified prompt-toolkit rendering channel (the other half of the original Phase E design) deferred to 0.6.0.

### Added

- `aura/core/tasks/` — `TasksStore` (keyed by `task_id`, append-only on completion), `SubagentFactory.spawn`, `run_task` background coroutine.
- `aura/tools/task_create.py` — stateful tool (Agent injects `store` + `factory` + `running` dict); `_arun` schedules `asyncio.create_task(run_task(...))` and returns `{task_id, description, status=running}` immediately. Done-callback pops the handle to prevent unbounded map growth.
- `aura/tools/task_output.py` — read-only, concurrency-safe; raises `ToolError` on unknown id.
- `aura/core/commands/tasks.py` — `/tasks` lists recent records (id prefix + status + description), sorted by `started_at` desc, top 20.
- `TaskRecord` dataclass: `{id, parent_id, description, prompt, status, messages, final_result, error, started_at, finished_at}`.

### Changed

- `Agent` now constructs a `TasksStore` + `SubagentFactory` at init and wires them into `task_create`. Uses pydantic `PrivateAttr` for `_running_tasks` (typed field would deep-copy the dict during validation and break identity-sharing with the tool).
- `Agent.close()` extended to cancel all in-flight `_running_tasks` and await their completion.

### Deferred (0.5.x / 0.6.0)

- Per-subagent permission inheritance.
- Recursion (subagent spawns subagent) — MVP strips `task_create` + `task_output` from the child's tool list by construction.
- Skill/MCP sharing with parent (child today gets `SkillRegistry()` and `mcp_servers=[]`).
- Full transcript accumulation (today only `Final` is recorded).
- Unified prompt-toolkit Application + foldable regions + dialog queue — `docs/specs/2026-04-21-phase-e-subagent.md` §Target architecture.

### Stats

- 903 → 921 tests. Lint + mypy clean (159 source files).

## [0.4.0] — Compact

Conversation compaction with state preservation. `Agent.compact(source=...)` summarizes history via an unbound summary model, preserves session state that can't be regenerated, and clears discovery caches that can.

### Headlines

- Manual `/compact` slash command; auto-trigger deferred to 0.4.1 (AgentLoop has no back-ref to Agent; adding one would violate the clean-loop invariant).
- Survives compact: `_read_records`, `_invoked_skills`, `state.custom['todos']`, session rules, `total_tokens_used`.
- Cleared + re-discovered: project_memory, rules cache, `_nested_fragments`, `_matched_rules`.
- `SystemMessage` untouched (prompt-cache friendly). Summary rendered as `<session-summary>` HumanMessage — same tag pattern as `<project-memory>` / `<skill-invoked>`.

### Added

- `aura/core/compact/` — `compact.py::run_compact` free function (complex logic stays independently testable); `prompt.py::SUMMARY_SYSTEM` enforces TEXT-ONLY + structured sections (goal / decisions / files-touched / tools-used / open-threads / next-steps).
- `Agent.compact(source="manual")` thin wrapper over `run_compact`.
- `/compact` command via `CompactCommand` (0.1.1 CommandRegistry).
- Constants pinned now so 0.4.1 auto-wiring doesn't shuffle module boundaries: `KEEP_LAST_N_TURNS=3`, `SUMMARY_BUDGET=50_000`, `AUTO_COMPACT_THRESHOLD=150_000` (declared, unused in 0.4.0).
- Journal event: `compact_applied` with `source` / `before_tokens` / `after_tokens`.

### Changed

- Summary turn dispatched via `self._model.ainvoke` directly (not `self._bound`) — the unbound model makes tool calls impossible by construction; "TEXT ONLY" prompt guard is the belt, the unbound model is the suspenders.
- `must_read_first` hook closure re-swapped post-compact (same pattern as `clear_session`) so the closure points at the new Context's `_read_records`.

### Deferred (0.4.1)

- Auto-trigger at `AUTO_COMPACT_THRESHOLD`.
- Selective re-injection of `MAX_FILES_TO_RESTORE=5` most-recent files with `MAX_TOKENS_PER_FILE=5000` cap (constants declared; today compact relies on the preserved tail to carry recent file contents).
- API-error reactive recompact (when the model returns "prompt too long").

### Stats

- 885 → 903 tests. Lint + mypy clean (148 source files).

## [0.3.0] — MCP via langchain-mcp-adapters

MCP client integration. A prior iteration handwrote transport + JSON-RPC correlation + JSON-Schema-to-pydantic bridge (~350 LOC). This release replaces that stack with `langchain-mcp-adapters` (`>=0.2,<1.0`); Aura's MCP module is purely the Aura-specific layer: metadata defaults, namespace prefix, command bridge, graceful degradation.

### Headlines

- `aura/core/mcp/` — `MCPManager` wraps `MultiServerMCPClient` with per-server error isolation.
- Tool namespace `mcp__<server>__<tool>` applied by Aura (not the library) if not already prefixed.
- Dynamic tool registration: `ToolRegistry.register()` / `unregister()`; `AgentLoop._rebind_tools()` re-binds the model with an updated tool set after post-construction discovery.
- `Agent.aconnect()` — async, post-construction; failures never re-raise (graceful degradation).

### Added

- `aura/core/mcp/adapter.py::add_aura_metadata` — attaches capability flags to library-returned tools. Defaults are conservative: `is_destructive=True`, `is_concurrency_safe=False`, `max_result_size_chars=30_000` (we don't trust unknown servers' read-only claims; users override per-server).
- `aura/core/mcp/adapter.py::make_mcp_command` — wraps an MCP prompt (fetched via `client.get_prompt`) as a slash command `/<server>__<prompt>` with `source="mcp"`, feeding through the 0.1.1 CommandRegistry.
- `aura/core/mcp/manager.py::MCPManager` — loops `get_tools(server_name=...)` per server (library's bulk `get_tools()` crashes on any single failure) and journals each failure as `mcp_connect_failed`. Defensive `stop_all` that reflects over `aclose` / `close` (library has no teardown API today).
- `AuraConfig.mcp_servers: list[MCPServerConfig]` — `{name, command, args, env, transport, enabled}`. stdio only (HTTP/SSE deferred to 0.3.x).
- `Agent.aconnect()` — establishes MCP connections, registers discovered tools, stores commands. No-op when no servers configured.
- CLI: `aura/cli/__main__.py` wraps REPL launch in an async entry that awaits `agent.aconnect()` before `run_repl_async`.

### Changed

- `ToolRegistry` gained `register(tool)` (raises on duplicate) and `unregister(name)` (idempotent — MCP disconnect safety).
- `AgentLoop.__init__` now saves `self._model` at init to support `_rebind_tools()`.
- `Agent.close()` extended to tear down MCP manager; handles both running-loop (`create_task`) and sync-close (`asyncio.run`) cases.
- `build_default_registry` extended to register `agent._mcp_commands` after skills.

### Known library limitations (documented, monitor upstream)

- No native per-server isolation in `get_tools()` — Aura loops.
- No teardown API — Aura reflects-probes for one.
- Prompt enumeration requires `async with client.session()` per server — `MCPManager` caches the list at connect time.

### Stats

- 868 → 885 tests. Lint + mypy clean (141 source files).

## [0.2.0] — Skills MVP (command-mode)

User-authored skills loaded from `~/.aura/skills/*.md` (global) and `<cwd>/.aura/skills/*.md` (project-local). Each skill registers as a slash-invokable command; its body is injected into context on invocation via Aura's existing tag-based Context mechanism — the equivalent of claude-code's `<system-reminder>` HumanMessage pattern.

### Headlines

- SKILL files: YAML frontmatter (`name` + `description` required) + markdown body.
- User layer wins on cross-layer name collision; intra-user-layer duplicates drop with journal warning (`skill_parse_failed` — matches `rules.py` observability pattern).
- Context gains `<skills-available>` (listing, rendered when non-empty) and `<skill-invoked name="...">` (body per invocation, deduped by source_path).
- Slash command per skill auto-prefixed `/<name>` via the 0.1.1 CommandRegistry.

### Added

- `aura/core/skills/` — `loader.py` (scans both layers, YAML frontmatter parse, non-recursive, `.md` only), `registry.py` (register / get / list), `command.py::SkillCommand` (`source="skill"`, handle calls `agent.record_skill_invocation`), `types.py::Skill`.
- Context fields: `_skills_available: list[Skill]` (populated at construction, rendered as one `<skills-available>` HumanMessage when non-empty — empty-tag bug avoided by omitting entirely), `_invoked_skills: list[Skill]` (append-only via `record_skill_invocation`).
- Placement in `build()` message list: between `<rule>` and `<todos>`. Rules predate skills in the cache prefix, so skills AFTER rules keeps cached prefixes stable as skills churn.
- `Agent.record_skill_invocation()` delegates to Context.
- `build_default_registry(agent=None)` extended: when `agent` is supplied, registers one `SkillCommand` per skill in `agent._skill_registry`; `/help` enumerates them automatically.

### Changed

- `Agent.__init__` loads skills via `load_skills(cwd=...)`, stores `_skill_registry`, threads skills into `_build_context(skills=...)`.
- `Agent.clear_session` rebuilds Context (invoked-skills list resets naturally via new instance); skills registry persists — same list is re-threaded into the new Context (no re-scan; hot reload is 0.2.x).
- REPL passes `agent` into `build_default_registry`.

### Deferred (0.2.x)

- Directory-form skills (`<dir>/SKILL.md` with supporting files).
- Triggers beyond slash command (file patterns, keyword match).
- LLM-invokable `SkillTool` (skill invocation from within a turn).
- Runtime file watcher for hot reload.
- Skills from MCP servers (depends on 0.3.0 which followed).

### Stats

- 842 → 868 tests. Lint + mypy clean (135 source files).

## [0.1.1] — Unified Command abstraction

Prereq release for Skills + MCP. Both need runtime command registration (skills expose slash commands; MCP prompts become commands); the old hardcoded if-else in `aura/cli/commands.py` couldn't accommodate either without editing that file every time.

### Headlines

- New `aura/core/commands/` subpackage: `Command` Protocol + `CommandRegistry`.
- Four builtins (`/help`, `/exit`, `/clear`, `/model`) migrated from hardcoded if-else to `Command` instances.
- `aura/cli/commands.py` shrunk to a ~50-line façade: `build_default_registry()` + `async dispatch(line, agent, registry)`.

### Added

- `aura/core/commands/types.py` — `Command` Protocol (`name` / `description` / `source` / async `handle`), `CommandResult` dataclass, `CommandKind` + `CommandSource` Literals. Protocol is NOT `@runtime_checkable` — mypy structural check at `register()` sites is sufficient and avoids isinstance overhead.
- `aura/core/commands/registry.py` — `CommandRegistry` with `register` / `unregister` / `list` / async `dispatch`. `register` raises on duplicate; `unregister` is idempotent (MCP disconnect safety); `list` returns sorted by name. `dispatch` returns `handled=False` for non-slash input, `handled=True` with `"unknown..."` for unknown slash.
- `aura/core/commands/builtin.py` — `HelpCommand` (holds registry ref for live enumeration), `ExitCommand`, `ClearCommand`, `ModelCommand` (preserves `_model_status` + `UnknownModelSpecError`).
- Forward-facing test `test_command_from_skill_source_registers_and_dispatches` — stub Command with `source="skill"` registers and dispatches cleanly, anchoring that the Protocol accommodates Skills/MCP without further churn.

### Changed

- **Breaking (internal)**: `dispatch(line, agent)` → `async dispatch(line, agent, registry)`. Only caller is the REPL — 3-line update (import + registry construction once at startup + await on dispatch).

### Stats

- 827 → 842 tests. Lint + mypy clean.

## [0.1.0] — Unreleased (walking skeleton MVP)

First functional MVP: own-loop agent with end-to-end permissions, 9 built-in tools, multi-layer context memory, hook-based extensibility, and claude-code-aligned safety invariants.

### Headlines

- Permissions subsystem end-to-end: rule-based access, per-tool matchers, decision audit, REPL prompt integration, atomic settings persistence.
- Tool safety floor: must-read-first invariant, bash Tier A checks (zsh-dangerous / CR / command-substitution / cd+git compound), 100 MB bash memory ceiling, SSRF-hardened web_fetch.
- Multi-layer context memory: 3-tier project discovery, `@import` with cycle + depth 5 limit, rules frontmatter, progressive nested-memory loading.
- Loop + hook architecture: `max_turns=50` cap (inline, not hook), `HookChain` with 4 lifecycle phases, async cancellation via `AbortSignal`-equivalent.
- Async REPL: survives mid-turn exceptions, rich markdown rendering, thinking spinner, slash commands.
- Config + LLM: JSON providers[] + router{} aliases, lazy SDK load, OpenAI / Anthropic / Ollama + any OpenAI-compatible endpoint.

### Added

**Permissions**
- `aura/core/permissions/` subpackage: `Decision` with closed-set `DecisionReason` + invariants, `Rule` parser + `RuleSet` + session-scoped `SessionRuleSet`, four `Mode` literals, `SafetyPolicy` with `DEFAULT_PROTECTED_WRITES` / `DEFAULT_PROTECTED_READS`, per-tool `rule_matcher` + `args_preview` factories, atomic `settings.json` + `settings.local.json` merge.
- `make_permission_hook` — allow / deny / ask via list-select CLI dialog; `Allow always` persists to session or project settings.
- `PermissionAudit` event, `rule_hint` auto-derivation from tool call.

**Safety floor**
- `aura/core/permissions/bash_safety.py` + `aura/core/hooks/bash_safety.py`: 4 Tier A checks, cannot be overridden by permission rules or `--bypass-permissions`.
- `aura/core/hooks/must_read_first.py` + `aura/core/memory/context.py::_read_records`: 4-way `read_status` (never_read / stale / partial / fresh) for `edit_file` + `write_file` overwrites.
- `web_fetch` SSRF checks (loopback, private-network, cloud-metadata IPs, DNS failure).

**Tools (9 built-in)**
- `read_file` with `offset` / `limit` + partial-read tracking.
- `write_file` with parent-dir auto-create.
- `edit_file` with must-read-first, mixed CRLF+LF per-line preservation, `old_str=""` new-file path, error-message echo for LLM self-correction.
- `grep` on ripgrep backend with 3 output modes (`files_with_matches` default / `content` / `count`), `-A`/`-B` context, multiline, `--type`, sentinel separators fixing `-<digits>-` path mis-parse.
- `glob` with mtime sort default (newest first) + alphabetical opt-in.
- `bash` natively async with `AbortSignal`-equivalent cancellation (SIGTERM → SIGKILL ladder), 30 KB per-stream output cap, 100 MB hard memory ceiling via streaming ring buffer.
- `todo_write` with single-`in_progress` cardinality validator.
- `ask_user_question` (free-text or 2–6 option list, TTY-gated).
- `web_fetch` with SSRF hardening + required permission prompt.

**Context & memory**
- `aura/core/memory/context.py`: 4-layer message assembly (system / eager primary / nested / history), progressive `on_tool_touched_path` nested-memory loading, dedup via `_loaded_nested_paths`.
- `aura/core/memory/project_memory.py`: 3-layer CLAUDE.md discovery + walk-up, `@import` with depth 5 + cycle detection + fence, session memoize + `clear_cache`.
- `aura/core/memory/rules.py`: frontmatter `paths` glob, per-tool-call trigger, journal observability on malformed rules (`rule_yaml_parse_failed` / `rule_paths_invalid_type` / `rule_glob_compile_failed`).

**Loop & hooks**
- `AgentLoop` with `max_turns=50` kwarg default; inline enforcement mirrors claude-code `query.ts:1705`.
- `HookChain` with 4 `Protocol`-typed phases (pre_model / post_model / pre_tool / post_tool).
- `Final(message, reason: Literal["natural", "max_turns"])` discriminator.

**CLI**
- `uv run aura` REPL: async, rich markdown, thinking spinner, Codara-style welcome banner.
- Slash commands: `/help`, `/exit`, `/clear`, `/model`.
- `--bypass-permissions` flag (disables permission prompts; banner re-announced every prompt); `--verbose` mode.

**Config & LLM**
- JSON `providers[]` + `router{}` — precedence: `$AURA_CONFIG` > project > user > defaults.
- Lazy SDK load (`openai` / `anthropic` / `ollama` extras); secret resolution with empty-string api_key → treated as missing.
- `ProviderConfig.params` pass-through for LangChain constructor kwargs.
- Protocol/Literal sync assertion (`tests/test_llm_create.py::test_protocols_dict_matches_provider_literal`).

**Audit & persistence**
- `aura/core/persistence/journal.py`: JSONL audit log, configurable target, silent no-op on unwritable parent, all safety decisions journalled (`permission_decision` / `safety_blocked` / `must_read_first_blocked` / `bash_safety_blocked` / `plaintext_api_key_warning` / `rule_*_failed`).
- SQLite session storage with transaction + rollback.

**Documentation**
- `docs/specs/2026-04-17-aura-mvp-design.md` — spec.
- `docs/specs/2026-04-21-phase-e-subagent.md` — deferred subagent + unified rendering channel design.
- `docs/research/claude-code-design-principles.md` — distilled study of claude-code's 5 subsystems.

### Changed

- **Breaking**: config format YAML → JSON with `providers[]` + `router{}`.
- **Breaking**: `AuraTool` protocol removed; tools now subclass LangChain `BaseTool` directly with `tool_metadata(...)`.
- **Breaking**: `grep` return shape changed to discriminated `{mode, ...}` (was `{matches, count, truncated}`).
- **Breaking**: `bash` return shape gained `truncated` and `killed_at_hard_ceiling` fields.
- **Breaking**: `web_fetch` no longer `is_read_only` — now prompts for permission.
- **Breaking**: `HookChain` Protocol classes (not Callable aliases) — hook factories must return typed hooks.
- Permission config moved to single entry (`AuraConfig.permissions` dropped).
- `make_max_turns_hook` / `MaxTurnsExceeded` removed — loop owns `max_turns` directly.
- `bash` migrated from sync `subprocess.run` to async `create_subprocess_shell` for structured cancellation.

### Fixed

- Must-read-first path appeared twice in `never_read` error message.
- Max-turns enforcement duplicated (hook + loop); now single source of truth.
- Bash command substitution bypass (`$(...)` / backticks / `eval` / `bash -c`) — added 5th Tier A rule.
- Bash output cap applied AFTER full buffering (DoS vector); now streams with ring buffer + 100 MB hard ceiling.
- Grep context-line parser mis-parsed paths with `-<digits>-` segments; fixed via `\x1f` / `\x02` sentinel separators.
- Edit_file mixed CRLF+LF files silently mutated bare-LF lines to CRLF; now detected + skips normalization.
- REPL crashed on mid-turn exceptions; now survives and prints error without leaking traceback.
- `clear_session` did not swap `must_read_first` hook closure to new Context (regression guard added).

### Deferred / known limitations

- **Subagent (`Task` tool) + unified rendering channel** — designed in `docs/specs/2026-04-21-phase-e-subagent.md`, not yet implemented. Awaits dedicated Phase E.
- **Token-budget diminishing-returns detection** — advisory feature; `max_turns=50` is the hard floor today.
- **MCP integration / skills / Tauri desktop** — not in MVP scope.
- **Bash command safety is Tier A only** — Tier B checks (IFS injection, ANSI-C quoting, obfuscated flags, non-safe redirections) deliberately flow through the permission system rather than the hard floor.

### Stats

- 171 commits, ~25 600 insertions / 273 deletions across 152 files.
- 827 tests passing (ruff + mypy clean on 121 source files).
- Platforms: macOS + Linux (Windows not supported — POSIX signal ladder in bash tool).
