"""Code-defined managed skills materialized into a hidden runtime root."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SKILL_FILE = "SKILL.md"

# Bundled skills materialize from code-defined content (below) into a
# hidden runtime root under ``~/.aura/plugins`` so the active catalogue is
# detached from the Python package layout.
_BUNDLED_SKILLS_EXTRACTED_ROOT_NAME = "skills"
_BUNDLED_CACHE_KEY = "aura-bundled-skills"
_bundled_skills_extraction: tuple[str, Path] | None = None
_BUNDLED_SKILL_FILES: dict[str, str] = {
    "verify": """---
description: Verify the most recent change works end-to-end before claiming done.
when_to_use: Before responding \"done\" / \"fixed\" / \"passing\" — run real checks.
---
# Verify

Before claiming a task is complete:

1. Run the project's tests (`make check`, `pytest`, `npm test`, etc.) and confirm
   they pass.
2. Re-run the specific failing case from the bug report — don't assume related
   tests cover it.
3. Read back the changed files to confirm the diff is what you intended.
4. If the change touches a CLI / API surface, exercise it end-to-end at least
   once instead of trusting unit tests alone.

Evidence before assertions: paste the actual command output that proves the
verification, not a paraphrase.
""",
    "simplify": """---
description: Review the diff for reuse, dead code, and over-engineering before commit.
when_to_use: After implementing a change, before committing — pause to simplify.
---
# Simplify

Pre-commit pass over the current diff:

1. Is there an existing helper / utility that already does this? Reuse it
   instead of duplicating.
2. Did you add a flag, knob, or abstraction that no caller currently exercises?
   Drop it — half-wired extensibility rots.
3. Are comments explaining \"what\" instead of \"why\"? Strip the \"what\"; the
   code shows what.
4. Is there dead code (unreachable branches, unused imports, stale docstrings
   referencing removed behavior)? Delete it.
5. Could the same outcome be expressed with fewer lines, fewer types, or one
   less indirection? Do it.

The bar: would a staff engineer approve this diff as-is, or would they ask
for one more pass? If the latter, do the pass now.
""",
    "code-review": """---
description: Code review the pending diff with explicit pass/fail criteria.
when_to_use: Before opening a PR or merging — surface real issues, not nits.
---
# Code review

Walk the diff with these checks. Surface only real issues; suppress nits.

## Correctness
- Does the code do what the description / spec / failing test says it should?
- Are edge cases handled (empty input, None, concurrent access, partial
  failure)?
- Are error paths tested or at least exercised by the new code?

## Safety
- New `subprocess`, `eval`, `pickle.loads`, raw SQL string concat, or shell
  interpolation? Check for injection.
- New file writes / deletes outside an obviously-bounded path?
- Secrets, tokens, internal hostnames in code or test fixtures?

## Maintainability
- Is the change minimal — only the lines that needed to change, changed?
- Public API additions: is each one used by a caller in this same diff? If
  not, defer them.
- New abstraction layers: is there a second concrete user, or is this YAGNI?

## Test quality
- New behavior has at least one test that would fail against `main`.
- Tests assert on observable behavior, not internal implementation details.
- No `# type: ignore`, `# noqa`, or `pytest.skip` added without a reason in
  the same line.

Pass criteria: every check above is satisfied. If any check fails, file the
issue against the diff before approving.
""",
}


@contextmanager
def _bundled_skills_root(*, home_dir: Path | None = None) -> Iterator[Path | None]:
    """Materialize bundled skills into a hidden runtime root; yield its path.

    Root: ``<home>/.aura/plugins/bundled-skills/<cache-key>/skills``. Cached
    across calls within one process; rebuilt on cache-key mismatch.
    """
    global _bundled_skills_extraction
    resolved_home = (home_dir or Path.home()).resolve()

    if _bundled_skills_extraction is not None:
        cached_key, cached_root = _bundled_skills_extraction
        if cached_key == _BUNDLED_CACHE_KEY and cached_root.is_dir():
            yield cached_root
            return
        if cached_root.exists():
            shutil.rmtree(cached_root)
        _bundled_skills_extraction = None

    extracted_root = (
        resolved_home / ".aura" / "plugins" / "bundled-skills"
        / _BUNDLED_CACHE_KEY / _BUNDLED_SKILLS_EXTRACTED_ROOT_NAME
    )
    if extracted_root.exists():
        shutil.rmtree(extracted_root)
    extracted_root.parent.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    for skill_name, body in _BUNDLED_SKILL_FILES.items():
        skill_dir = extracted_root / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / _SKILL_FILE).write_text(body, encoding="utf-8")
    _bundled_skills_extraction = (_BUNDLED_CACHE_KEY, extracted_root)
    yield extracted_root
