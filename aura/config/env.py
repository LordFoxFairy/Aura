"""Environment-variable expansion for config values."""

from __future__ import annotations

import os
import re

_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env_vars(text: str, *, _missing_log: list[str] | None = None) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references in *text*.

    Output is NOT recursively re-expanded; empty env values count as missing.
    """
    if "${" not in text:
        return text

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        val = os.environ.get(name, "")
        if val:
            return val
        if default is not None:
            return default
        if _missing_log is not None and name not in _missing_log:
            _missing_log.append(name)
        return ""

    return _PATTERN.sub(_sub, text)
