"""User-scope approval store for project-layer MCP servers.

A project-layer ``mcp_servers.json`` (checked into a repo's ``.aura/``
directory) is an RCE channel: the file can spawn arbitrary subprocesses
on first ``aura`` invocation. Mirroring claude-code's
``enabledMcpjsonServers`` pattern, we gate every project-layer server
behind an explicit per-server approval that persists in **user scope**
(``~/.aura/mcp-approvals.json``) keyed by ``(project_path, server_name)``.

User-scope ``mcp_servers.json`` (``~/.aura/mcp_servers.json``) is NOT
gated — that file came from the user's own ``aura mcp add`` and is
already authoritative. Only project-layer entries hit this code path.

Schema (v1)
-----------
::

    {
      "version": 1,
      "approvals": {
        "<project_path_abs>": {
          "<server_name>": {
            "fingerprint": "<sha256-hex>",
            "approved_at": "<iso8601-utc>"
          }
        }
      }
    }

The fingerprint covers ``command + args + sorted env keys``. Env *values*
are intentionally excluded so a token rotation does not invalidate the
approval, but any change to the command line (e.g. swapping ``npx
some-package`` for ``curl ... | bash``) re-prompts. The set of env keys
is included because adding a new env key changes the server's effective
behaviour even if values rotate.

Concurrency
-----------
Writes go through ``os.replace`` after a temp file is fully flushed +
fsync'd, so a crash mid-write leaves the previous file intact. Reads
are best-effort: a malformed approvals file logs to journal and
returns empty (treats every project server as un-approved). This is
the conservative direction: we'd rather re-prompt than silently load a
poisoned file.
"""

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

_APPROVALS_FILENAME = "mcp-approvals.json"
_SCHEMA_VERSION = 1


def approvals_path() -> Path:
    """Return ``~/.aura/mcp-approvals.json`` (expanded, may not exist).

    Co-located with ``mcp_servers.json`` (also under ``~/.aura/``) so a
    user wiping ``~/.aura/`` resets both store and approvals together.
    """
    return Path.home() / ".aura" / _APPROVALS_FILENAME


def project_key(cwd: Path | None = None) -> str:
    """Return the canonical project-path key for the approvals store.

    Resolves symlinks so a project at ``~/work/foo`` reached via
    ``/Volumes/dev/foo`` (a symlink) maps to the same key. Falls back to
    the absolute (non-resolved) path if resolution fails — better to
    re-prompt occasionally than to error out.
    """
    base = cwd if cwd is not None else Path.cwd()
    try:
        return str(base.resolve())
    except OSError:
        return str(base.absolute())


def fingerprint(cfg: MCPServerConfig) -> str:
    """Compute a stable fingerprint of *cfg*'s execution-relevant fields.

    For stdio: ``command``, every ``args[i]``, and the *sorted set of
    env keys* (not values). Env values are excluded because rotating a
    secret should not force re-approval; env *keys* are included
    because adding a new env input meaningfully changes server
    behaviour even at constant values.

    For sse / streamable_http: ``url`` and the *sorted set of header
    names* (not values, same rationale).

    The transport itself is part of the fingerprint so flipping a
    server from ``stdio`` to ``streamable_http`` (or vice versa)
    re-prompts.
    """
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
    """Read and parse the approvals file; return ``{}`` on any error.

    A missing file is the first-run path. A malformed file (bad JSON,
    wrong shape) is logged via :mod:`aura.core.persistence.journal` and
    treated as empty — silently re-prompting is safer than honouring a
    half-readable approvals file. Callers always see a dict shape.
    """
    path = approvals_path()
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        try:
            from aura.core import journal  # local to avoid import cycle
            journal.write(
                "mcp_approvals_load_failed",
                path=str(path),
            )
        except Exception:  # noqa: BLE001
            pass
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _normalise(raw: dict[str, Any]) -> dict[str, dict[str, _Approval]]:
    """Coerce the on-disk shape into ``{project: {server: _Approval}}``.

    Forward-compatible: unknown top-level keys are ignored, malformed
    entries are dropped (not raised), so an operator hand-editing the
    file can't take the agent down.
    """
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
    """Return ``{server_name: _Approval}`` for the named project.

    ``project`` defaults to :func:`project_key` (i.e., ``Path.cwd()``).
    Missing project entry → empty dict (no error).
    """
    key = project if project is not None else project_key()
    return _normalise(_load_raw()).get(key, {})


def is_approved(cfg: MCPServerConfig, *, project: str | None = None) -> bool:
    """Return True iff *cfg* has a current, fingerprint-matching approval.

    "Current" means: an entry exists for ``(project, cfg.name)`` AND
    the stored fingerprint equals the one we'd compute now from
    *cfg*. A mismatch indicates the server's config changed since
    approval — the user must re-approve.
    """
    bucket = load_for_project(project=project)
    entry = bucket.get(cfg.name)
    if entry is None:
        return False
    return entry.fingerprint == fingerprint(cfg)


def approval_state(
    cfg: MCPServerConfig, *, project: str | None = None,
) -> str:
    """Return ``"approved"`` / ``"changed"`` / ``"unapproved"`` for *cfg*.

    Tristate so the caller can distinguish "never seen" (cold first
    run) from "approved earlier but config changed since" (stale
    approval — UX hint should mention the diff).
    """
    bucket = load_for_project(project=project)
    entry = bucket.get(cfg.name)
    if entry is None:
        return "unapproved"
    if entry.fingerprint != fingerprint(cfg):
        return "changed"
    return "approved"


def _atomic_write(payload: dict[str, Any]) -> None:
    """Write ``payload`` to the approvals file via temp file + rename.

    ``os.replace`` is atomic on POSIX and Windows (since Python 3.3);
    a concurrent reader either sees the old file or the new one,
    never a partial write. The temp file lives in the same parent
    directory so the rename never crosses filesystems.
    """
    path = approvals_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``delete=False`` so we can rename out from under the context manager
    # without the close() call deleting our destination.
    fd, tmp_name = tempfile.mkstemp(
        prefix=".mcp-approvals.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            # tmpfs / network mounts may reject fsync — same policy
            # as the journal module (best-effort durability).
            with contextlib.suppress(OSError):
                os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the temp file on any failure so we don't leak debris.
        with _suppress_oserror():
            os.unlink(tmp_name)
        raise


class _suppress_oserror:
    """Context manager that swallows :class:`OSError` (file-cleanup helper).

    Inlined to avoid a top-level import cycle on
    :mod:`contextlib` in this small module.
    """

    def __enter__(self) -> _suppress_oserror:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)


def approve(
    cfg: MCPServerConfig, *, project: str | None = None,
) -> None:
    """Persist an approval for *cfg* under *project*.

    Idempotent: re-approving an already-approved server just refreshes
    the fingerprint + timestamp (which is what the user wants if their
    config drifted and they're consciously re-approving it).

    Atomic: see :func:`_atomic_write`. Concurrent
    ``approve(serverA)`` + ``approve(serverB)`` calls may produce a
    last-writer-wins result (the second write reads the original file,
    not the in-flight one), but neither is corrupted. Aura is
    single-process for the CLI / REPL, so this matters only for tests.
    """
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
    """Remove the approval for ``name`` under *project*.

    Returns True if an approval was actually removed, False if no
    matching entry existed (idempotent — caller can ignore the return
    value if it doesn't care about distinguishing the cases).
    """
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
        # Drop the project bucket entirely once empty so the file
        # doesn't accumulate stale project entries forever.
        del approvals[key]
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
