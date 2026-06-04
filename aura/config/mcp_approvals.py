"""User-scope approval store gating project-layer MCP servers against RCE."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aura.config.schema import MCPServerConfig
from aura.infrastructure.persistence import journal

_APPROVALS_FILENAME = "mcp-approvals.json"
_SCHEMA_VERSION = 1


def approvals_path() -> Path:
    return Path.home() / ".aura" / _APPROVALS_FILENAME


def project_key(cwd: Path | None = None) -> str:
    # Resolve symlinks so the same dir reached via two paths maps identically.
    base = cwd if cwd is not None else Path.cwd()
    try:
        return str(base.resolve())
    except OSError:
        return str(base.absolute())


def fingerprint(cfg: MCPServerConfig) -> str:
    h = hashlib.sha256()
    h.update(cfg.transport.encode("utf-8"))
    h.update(b"\x00")
    if cfg.transport == "stdio":
        h.update((cfg.command or "").encode("utf-8"))
        h.update(b"\x00")
        for arg in cfg.args:
            h.update(arg.encode("utf-8"))
            h.update(b"\x00")
        for env_key in sorted(cfg.env.keys()):
            h.update(env_key.encode("utf-8"))
            h.update(b"\x00")
    else:
        h.update((cfg.url or "").encode("utf-8"))
        h.update(b"\x00")
        for hdr_key in sorted(cfg.headers.keys()):
            h.update(hdr_key.encode("utf-8"))
            h.update(b"\x00")
    return h.hexdigest()


@dataclass(frozen=True)
class _Approval:
    fingerprint: str
    approved_at: str


def _load_raw() -> dict[str, Any]:
    path = approvals_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        with contextlib.suppress(Exception):
            journal.write("mcp_approvals_load_failed", path=str(path))
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _normalise(raw: dict[str, Any]) -> dict[str, dict[str, _Approval]]:
    # Drop malformed entries so hand-edits can't take the agent down.
    out: dict[str, dict[str, _Approval]] = {}
    approvals = raw.get("approvals")
    if not isinstance(approvals, dict):
        return out
    for project, servers in approvals.items():
        if not isinstance(project, str) or not isinstance(servers, dict):
            continue
        bucket: dict[str, _Approval] = {}
        for name, entry in servers.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                continue
            fp = entry.get("fingerprint")
            ts = entry.get("approved_at")
            if not isinstance(fp, str) or not isinstance(ts, str):
                continue
            bucket[name] = _Approval(fingerprint=fp, approved_at=ts)
        if bucket:
            out[project] = bucket
    return out


def load_for_project(project: str | None = None) -> dict[str, _Approval]:
    key = project if project is not None else project_key()
    return _normalise(_load_raw()).get(key, {})


def is_approved(cfg: MCPServerConfig, *, project: str | None = None) -> bool:
    bucket = load_for_project(project=project)
    entry = bucket.get(cfg.name)
    if entry is None:
        return False
    return entry.fingerprint == fingerprint(cfg)


def approval_state(
    cfg: MCPServerConfig,
    *,
    project: str | None = None,
) -> str:
    """Tristate so callers distinguish cold first-run from drifted approval."""
    bucket = load_for_project(project=project)
    entry = bucket.get(cfg.name)
    if entry is None:
        return "unapproved"
    if entry.fingerprint != fingerprint(cfg):
        return "changed"
    return "approved"


def _atomic_write(payload: dict[str, Any]) -> None:
    path = approvals_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False: we rename the temp out from under the fd before close.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".mcp-approvals.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            # tmpfs / NFS may reject fsync; durability is best-effort.
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def approve(
    cfg: MCPServerConfig,
    *,
    project: str | None = None,
) -> None:
    # Idempotent: re-approving refreshes fingerprint + timestamp.
    raw = _load_raw()
    approvals = raw.get("approvals")
    if not isinstance(approvals, dict):
        approvals = {}
    key = project if project is not None else project_key()
    bucket = approvals.get(key)
    if not isinstance(bucket, dict):
        bucket = {}
    bucket[cfg.name] = {
        "fingerprint": fingerprint(cfg),
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    approvals[key] = bucket
    payload = {
        "version": _SCHEMA_VERSION,
        "approvals": approvals,
    }
    _atomic_write(payload)


def revoke(name: str, *, project: str | None = None) -> bool:
    # Returns True iff an entry was removed (idempotent on missing entry).
    raw = _load_raw()
    approvals = raw.get("approvals")
    if not isinstance(approvals, dict):
        return False
    key = project if project is not None else project_key()
    bucket = approvals.get(key)
    if not isinstance(bucket, dict) or name not in bucket:
        return False
    del bucket[name]
    if not bucket:
        del approvals[key]  # drop empty bucket so the file doesn't accumulate stale projects
    payload = {
        "version": _SCHEMA_VERSION,
        "approvals": approvals,
    }
    _atomic_write(payload)
    return True


__all__ = [
    "approval_state",
    "approvals_path",
    "approve",
    "fingerprint",
    "is_approved",
    "load_for_project",
    "project_key",
    "revoke",
]
