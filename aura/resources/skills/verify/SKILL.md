---
description: Verify the most recent change works end-to-end before claiming done.
when_to_use: Before responding "done" / "fixed" / "passing" — run real checks.
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
