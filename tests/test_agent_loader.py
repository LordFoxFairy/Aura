"""Filesystem-loaded subagent definitions.

Covers :mod:`aura.infrastructure.agents.loader`: built-ins always load,
markdown files with YAML frontmatter parse into ``AgentDef``s, missing
``.aura/agents`` directory falls back to built-ins only.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aura.infrastructure.agents import (
    AgentDef,
    all_agent_defs,
    builtin_agents,
    get_agent_def,
    load_agents,
)
from aura.infrastructure.agents import loader as _loader
from aura.infrastructure.persistence import journal


def test_load_agents_returns_builtins_when_no_dir(tmp_path: Path) -> None:
    # Empty project root → loader sees no ``.aura/agents/`` and falls
    # straight through to the built-in registry; the four built-in
    # names must all be present.
    out = load_agents(tmp_path)
    names = set(out.keys())
    assert {"general-purpose", "explore", "verify", "plan"} <= names
    # Built-ins are AgentDef instances (not the legacy AgentTypeDef
    # shape) — guards against an accidental re-import after the rename.
    for d in out.values():
        assert isinstance(d, AgentDef)


def test_load_agents_picks_up_filesystem_md(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "research.md").write_text(
        "---\n"
        "name: research\n"
        "description: Deep-research subagent for long-context lookups.\n"
        "tools: [read_file, grep, glob, web_fetch, web_search]\n"
        "---\n"
        "You are a **Research** subagent. Cite sources.\n",
        encoding="utf-8",
    )
    out = load_agents(tmp_path)
    assert "research" in out
    research = out["research"]
    assert research.name == "research"
    assert "Deep-research" in research.description
    assert {"read_file", "grep", "glob", "web_fetch", "web_search"} <= research.tools
    assert "Research" in research.system_prompt_suffix


def test_filesystem_overrides_builtin(tmp_path: Path) -> None:
    # A user-defined ``explore.md`` must REPLACE the built-in explore
    # def (not merge with it). Matches claude-code's loader precedence:
    # filesystem wins.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "explore.md").write_text(
        "---\n"
        "name: explore\n"
        "description: Custom explore — tailored to this project.\n"
        "tools: [read_file]\n"
        "---\n"
        "Custom prompt body.\n",
        encoding="utf-8",
    )
    out = load_agents(tmp_path)
    explore = out["explore"]
    assert "Custom explore" in explore.description
    assert explore.tools == frozenset({"read_file"})


def test_missing_required_field_skips_file(tmp_path: Path) -> None:
    # File without a ``description`` is silently dropped (journaled) —
    # built-ins still load, so a typoed user file doesn't strand the
    # whole loader.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "broken.md").write_text(
        "---\nname: broken\n---\nNo description provided.\n",
        encoding="utf-8",
    )
    out = load_agents(tmp_path)
    assert "broken" not in out
    assert "general-purpose" in out


def test_all_agent_defs_leads_with_builtins(tmp_path: Path) -> None:
    # Even with user-defined agents present, the four built-ins must
    # appear first in declaration order so the LLM-facing catalogue
    # surfaces the defaults before the project-specific extras.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "zeta.md").write_text(
        "---\nname: zeta\ndescription: Zeta subagent.\n---\nBody.\n",
        encoding="utf-8",
    )
    defs = all_agent_defs(tmp_path)
    names = [d.name for d in defs]
    assert names[:4] == ["general-purpose", "explore", "verify", "plan"]
    assert "zeta" in names


def test_builtin_agents_returns_fresh_dict() -> None:
    # ``builtin_agents`` must hand back a fresh dict so mutations don't
    # pin onto the shared registry.
    a = builtin_agents()
    b = builtin_agents()
    assert a == b
    a.pop("explore")
    assert "explore" in builtin_agents()


@pytest.fixture(autouse=True)
def _reset_journal() -> Iterator[None]:
    # The loader journals skip/error events through the module-global
    # ``journal._path``; reset around every test so a configured sink in
    # one test can never leak into the next and corrupt later assertions.
    journal.reset()
    yield
    journal.reset()


def _make_agent_file(tmp_path: Path, filename: str, content: str) -> Path:
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def _journal_events(journal_path: Path) -> list[dict[str, object]]:
    if not journal_path.exists():
        return []
    return [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_no_frontmatter_uses_stem_and_skips_without_description(
    tmp_path: Path,
) -> None:
    # A file with no YAML frontmatter at all yields an empty mapping, so
    # ``name`` falls back to the stem but ``description`` stays empty —
    # the file must be dropped, never registered with a blank description.
    _make_agent_file(tmp_path, "plain.md", "just a body, no fences\n")
    out = load_agents(tmp_path)
    assert "plain" not in out
    assert "general-purpose" in out


def test_scalar_frontmatter_is_ignored_not_crashed(tmp_path: Path) -> None:
    # YAML that parses to a bare scalar (not a mapping) must not raise an
    # AttributeError on ``.get``; the loader treats it as no frontmatter
    # and skips the file for want of a description.
    _make_agent_file(tmp_path, "scalar.md", "---\njust a string\n---\nbody\n")
    out = load_agents(tmp_path)
    assert "scalar" not in out
    assert {"general-purpose", "explore", "verify", "plan"} <= set(out)


def test_list_frontmatter_is_ignored_not_crashed(tmp_path: Path) -> None:
    # YAML that parses to a list (sequence, not mapping) is likewise
    # rejected as non-dict frontmatter rather than crashing the loader.
    _make_agent_file(tmp_path, "seq.md", "---\n- a\n- b\n---\nbody\n")
    out = load_agents(tmp_path)
    assert "seq" not in out
    assert "general-purpose" in out


def test_empty_frontmatter_block_falls_back_to_builtins(tmp_path: Path) -> None:
    # An empty YAML block (``yaml.safe_load`` → ``None``) must coerce to an
    # empty dict, not blow up on ``None.get``; the file is dropped.
    _make_agent_file(tmp_path, "empty.md", "---\n\n---\nbody\n")
    out = load_agents(tmp_path)
    assert "empty" not in out
    assert "general-purpose" in out


def test_name_falls_back_to_file_stem(tmp_path: Path) -> None:
    # When frontmatter omits ``name`` but supplies a description, the file
    # stem is the stable identifier — keeps ``research.md`` addressable as
    # ``research`` without forcing redundant ``name:`` duplication.
    _make_agent_file(
        tmp_path,
        "stemname.md",
        "---\ndescription: Only a description here.\n---\nBody.\n",
    )
    out = load_agents(tmp_path)
    assert "stemname" in out
    assert out["stemname"].description == "Only a description here."


def test_blank_name_string_is_rejected(tmp_path: Path) -> None:
    # An explicit but whitespace-only ``name`` strips to empty; the file is
    # dropped rather than registered under the empty-string key, which would
    # be unaddressable by ``get_agent_def``.
    _make_agent_file(
        tmp_path,
        "blank.md",
        "---\nname: '   '\ndescription: has desc\n---\nbody\n",
    )
    out = load_agents(tmp_path)
    assert "" not in out
    # Stem fallback does NOT apply here: ``name`` is present-but-blank, so
    # the loader keeps the blank value and the missing-name guard fires.
    assert "blank" not in out


def test_tools_comma_string_is_split_and_trimmed(tmp_path: Path) -> None:
    # Authors may write ``tools`` as a CSV string; it must split on commas,
    # trim whitespace, and drop empties so a trailing/double comma can't
    # inject a phantom empty tool name into the allowlist.
    _make_agent_file(
        tmp_path,
        "csv.md",
        "---\nname: csv\ndescription: d\ntools: read_file, grep ,, glob\n---\nbody\n",
    )
    out = load_agents(tmp_path)
    assert out["csv"].tools == frozenset({"read_file", "grep", "glob"})


def test_tools_non_list_scalar_coerces_to_empty(tmp_path: Path) -> None:
    # A malformed ``tools: 42`` (neither list nor str) must coerce to the
    # empty frozenset — the "inherit all" sentinel — instead of crashing on
    # iteration over an int.
    _make_agent_file(
        tmp_path,
        "badtools.md",
        "---\nname: badtools\ndescription: d\ntools: 42\n---\nbody\n",
    )
    out = load_agents(tmp_path)
    assert out["badtools"].tools == frozenset()


def test_tools_numeric_entries_are_stringified(tmp_path: Path) -> None:
    # YAML list entries that parse as ints (``tools: [1, 2]``) are coerced
    # to their string form so the allowlist stays a ``frozenset[str]``.
    _make_agent_file(
        tmp_path,
        "numtools.md",
        "---\nname: numtools\ndescription: d\ntools: [1, 2]\n---\nbody\n",
    )
    out = load_agents(tmp_path)
    assert out["numtools"].tools == frozenset({"1", "2"})


def test_missing_fields_event_is_journaled(tmp_path: Path) -> None:
    # The skip must be auditable: a file missing its description journals an
    # ``agent_loader_missing_fields`` event carrying the path and which
    # fields were present, so a typo is diagnosable without re-running.
    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    path = _make_agent_file(tmp_path, "nodesc.md", "---\nname: nodesc\n---\nbody\n")
    load_agents(tmp_path)
    events = _journal_events(journal_path)
    missing = [e for e in events if e["event"] == "agent_loader_missing_fields"]
    assert len(missing) == 1
    assert missing[0]["path"] == str(path)
    assert missing[0]["has_name"] is True
    assert missing[0]["has_description"] is False


def test_unreadable_file_is_journaled_and_skipped(tmp_path: Path) -> None:
    # A ``*.md`` path that is actually a directory makes ``read_text`` raise
    # an OSError; the loader must journal ``agent_loader_read_error`` and
    # skip it, never propagate the OSError up through ``load_agents``.
    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "isdir.md").mkdir()
    out = load_agents(tmp_path)
    assert "general-purpose" in out
    events = _journal_events(journal_path)
    read_errs = [e for e in events if e["event"] == "agent_loader_read_error"]
    assert len(read_errs) == 1
    assert "IsADirectoryError" in str(read_errs[0]["error"])


def test_glob_oserror_falls_back_to_builtins_and_journals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If enumerating the agents directory raises (e.g. permission denied),
    # the loader must degrade to built-ins only and journal
    # ``agent_loader_dir_error`` — a broken project dir can't strand the
    # whole subagent catalogue.
    journal_path = tmp_path / "audit.jsonl"
    journal.configure(journal_path)
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)

    def _boom(self: Path, pattern: str) -> list[Path]:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "glob", _boom)
    out = load_agents(tmp_path)
    assert {"general-purpose", "explore", "verify", "plan"} <= set(out)
    events = _journal_events(journal_path)
    dir_errs = [e for e in events if e["event"] == "agent_loader_dir_error"]
    assert len(dir_errs) == 1
    assert "PermissionError" in str(dir_errs[0]["error"])


def test_body_is_stripped_of_leading_trailing_newlines(tmp_path: Path) -> None:
    # The system-prompt suffix is appended verbatim to the parent prompt, so
    # stray blank lines around the body must be trimmed to avoid injecting
    # ragged whitespace into the composed prompt.
    _make_agent_file(
        tmp_path,
        "body.md",
        "---\nname: body\ndescription: d\n---\n\n\nprompt text\n\n",
    )
    out = load_agents(tmp_path)
    assert out["body"].system_prompt_suffix == "prompt text"


def test_load_agents_is_idempotent(tmp_path: Path) -> None:
    # Loading twice from the same tree must yield identical registries —
    # the loader holds no mutable cross-call state, so repeated discovery
    # never duplicates or drifts the agent set.
    _make_agent_file(
        tmp_path,
        "dup.md",
        "---\nname: dup\ndescription: d\ntools: [read_file]\n---\nbody\n",
    )
    first = load_agents(tmp_path)
    second = load_agents(tmp_path)
    assert first == second
    assert first["dup"] == second["dup"]


def test_filesystem_override_is_not_mutated_across_calls(tmp_path: Path) -> None:
    # Mutating the dict returned by one ``load_agents`` call must not leak
    # into the next — each call rebuilds from a fresh ``builtin_agents``
    # copy, so the shared built-in registry stays pristine.
    first = load_agents(tmp_path)
    first.pop("explore")
    second = load_agents(tmp_path)
    assert "explore" in second


def test_get_agent_def_returns_builtin(tmp_path: Path) -> None:
    # Happy-path lookup resolves a built-in by name from an empty project
    # tree, confirming ``get_agent_def`` rides on the same merged registry.
    found = get_agent_def("explore", tmp_path)
    assert isinstance(found, AgentDef)
    assert found.name == "explore"


def test_get_agent_def_unknown_lists_valid_names(tmp_path: Path) -> None:
    # An unknown ``agent_type`` must raise with the sorted valid-name list
    # embedded so the LLM can self-correct from the error text alone.
    _make_agent_file(tmp_path, "zeta.md", "---\nname: zeta\ndescription: z\n---\nb\n")
    with pytest.raises(ValueError) as excinfo:
        get_agent_def("does-not-exist", tmp_path)
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "explore" in message
    assert "zeta" in message


def test_get_agent_def_resolves_filesystem_override(tmp_path: Path) -> None:
    # A filesystem ``explore.md`` must be the def returned by name lookup,
    # proving override precedence flows through ``get_agent_def`` and not
    # just the raw ``load_agents`` dict.
    _make_agent_file(
        tmp_path,
        "explore.md",
        "---\nname: explore\ndescription: Custom one.\ntools: [grep]\n---\nb\n",
    )
    found = get_agent_def("explore", tmp_path)
    assert found.description == "Custom one."
    assert found.tools == frozenset({"grep"})


def test_all_agent_defs_sorts_extras_after_builtins(tmp_path: Path) -> None:
    # Built-ins lead in declaration order; project extras follow in sorted
    # filename order so the LLM-facing catalogue is stable and predictable
    # regardless of directory iteration order.
    _make_agent_file(tmp_path, "beta.md", "---\nname: beta\ndescription: b\n---\nx\n")
    _make_agent_file(tmp_path, "alpha.md", "---\nname: alpha\ndescription: a\n---\nx\n")
    defs = all_agent_defs(tmp_path)
    names = [d.name for d in defs]
    assert names[:4] == ["general-purpose", "explore", "verify", "plan"]
    assert names[4:] == ["alpha", "beta"]


def test_parse_frontmatter_returns_full_text_when_no_fences() -> None:
    # The frontmatter parser must hand back the original text untouched when
    # no fences are present, so a bare-body file's content is preserved for
    # the stem-named fallback path rather than silently emptied.
    fm, body = _loader._parse_frontmatter("no fences here\nsecond line\n")
    assert fm == {}
    assert body == "no fences here\nsecond line\n"


def test_dir_without_md_files_falls_back_to_builtins(tmp_path: Path) -> None:
    # An existing ``.aura/agents`` dir containing only non-``.md`` files
    # globs to nothing; the loop body never runs and built-ins pass through
    # unchanged.
    agents_dir = tmp_path / ".aura" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    out = load_agents(tmp_path)
    assert {"general-purpose", "explore", "verify", "plan"} <= set(out)
    assert len(out) == 4


def test_load_agents_accepts_str_cwd(tmp_path: Path) -> None:
    # ``cwd`` may arrive as a ``str`` (CLI argv) or ``Path``; both must
    # resolve to the same registry so the public API is path-type agnostic.
    _make_agent_file(tmp_path, "sigma.md", "---\nname: sigma\ndescription: s\n---\nb\n")
    from_str = load_agents(str(tmp_path))
    from_path = load_agents(tmp_path)
    assert "sigma" in from_str
    assert from_str == from_path
