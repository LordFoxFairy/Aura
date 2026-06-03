"""Quote-aware lexical segmentation for bash-safety rules."""

from __future__ import annotations

import shlex


def _split_segments_quote_aware(command: str) -> list[str]:
    """Split on ``;``/``&&``/``||``/``|``/newline outside quoted regions.
    Needed so ``echo "rm -rf" > /tmp/x`` stays one segment."""
    segments: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]

        if ch == "\\" and not in_single and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            two = command[i:i + 2]
            if two in ("&&", "||"):
                segments.append("".join(buf))
                buf = []
                i += 2
                continue
            if ch in (";", "|", "\n"):
                segments.append("".join(buf))
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    segments.append("".join(buf))
    return segments


def _pipe_segments_quote_aware(command: str) -> list[str]:
    """Pipeline-only split — ``|`` outside quotes, NOT ``||``. Needed so
    pipe-to-shell sees only true pipeline boundaries."""
    segments: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]

        if ch == "\\" and not in_single and i + 1 < n:
            buf.append(ch)
            buf.append(command[i + 1])
            i += 2
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue

        if not in_single and not in_double:
            if command[i:i + 2] == "||":
                buf.append("||")
                i += 2
                continue
            if ch == "|":
                segments.append("".join(buf))
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    segments.append("".join(buf))
    return segments


def _expand_braces(text: str) -> list[str]:
    """Expand ``{a,b,c}`` brace lists (nested OK; sequence ``{1..10}`` not)
    so ``rm -rf /{etc,tmp}`` flags both targets at the safety check."""
    start = text.find("{")
    if start == -1:
        return [text]
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return [text]
    prefix = text[:start]
    suffix = text[end + 1:]
    body = text[start + 1: end]
    parts: list[str] = []
    depth = 0
    cursor = 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[cursor:i])
            cursor = i + 1
    parts.append(body[cursor:])
    if len(parts) <= 1:
        return [text]
    out: list[str] = []
    suffix_expansions = _expand_braces(suffix)
    for p in parts:
        for sub in _expand_braces(p):
            for suf in suffix_expansions:
                out.append(prefix + sub + suf)
    return out


def _first_token(segment: str) -> str | None:
    """First shlex token after leading ``VAR=value`` assignments."""
    stripped = segment.strip()
    if not stripped:
        return None
    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    for tok in tokens:
        if "=" in tok and not tok.startswith("=") and tok[0].isalpha():
            lhs = tok.split("=", 1)[0]
            if lhs.replace("_", "").isalnum():
                continue
        return tok
    return None
