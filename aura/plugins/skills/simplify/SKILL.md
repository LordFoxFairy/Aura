---
description: Review the diff for reuse, dead code, and over-engineering before commit.
when_to_use: After implementing a change, before committing — pause to simplify.
---
# Simplify

Pre-commit pass over the current diff:

1. Is there an existing helper / utility that already does this? Reuse it
   instead of duplicating.
2. Did you add a flag, knob, or abstraction that no caller currently exercises?
   Drop it — half-wired extensibility rots.
3. Are comments explaining "what" instead of "why"? Strip the "what"; the
   code shows what.
4. Is there dead code (unreachable branches, unused imports, stale docstrings
   referencing removed behavior)? Delete it.
5. Could the same outcome be expressed with fewer lines, fewer types, or one
   less indirection? Do it.

The bar: would a staff engineer approve this diff as-is, or would they ask
for one more pass? If the latter, do the pass now.
