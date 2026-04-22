# Changelog

Notable changes to Aura. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

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
