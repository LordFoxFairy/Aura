---
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
