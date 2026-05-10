"""Phase 1 Task 11 — :class:`Compactor` Protocol conformance.

Verifies that :class:`aura.core.compact.legacy_adapter.LegacyCompactor`
satisfies the :class:`aura.core.compact.Compactor` Protocol both via
``runtime_checkable`` ``isinstance`` *and* via attribute-level shape
checks. The Protocol is async on every method; ``isinstance`` only
checks for attribute presence, so we additionally confirm each method
exists, is callable, and is async.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from aura.core.compact import Compactor
from aura.core.compact.legacy_adapter import LegacyCompactor


def test_legacy_compactor_is_protocol_instance() -> None:
    """Class itself satisfies the runtime-checkable Protocol shape.

    ``Compactor`` is decorated ``@runtime_checkable`` so an
    ``isinstance(obj, Compactor)`` check confirms the three required
    method names (``microcompact`` / ``reactive`` / ``auto``) are
    present as attributes on the instance.

    A bare class instance can't be constructed without an Agent, but
    Protocol membership is structural, so we test against the class
    via a minimal ``object.__new__`` shell — that's enough for the
    isinstance check (which only walks ``__dict__``).
    """
    instance = object.__new__(LegacyCompactor)
    assert isinstance(instance, Compactor)


def test_legacy_compactor_methods_are_async() -> None:
    """All three Compactor methods are coroutines per spec §3.3."""
    for name in ("microcompact", "reactive", "auto"):
        method = getattr(LegacyCompactor, name)
        assert callable(method), f"{name} must be callable"
        assert inspect.iscoroutinefunction(method), (
            f"{name} must be ``async def`` per Compactor Protocol"
        )


def test_legacy_compactor_signatures_match_protocol() -> None:
    """Argument names match the Protocol declaration in ``__init__.py``.

    Spec §3.3 fixes the keyword names (``messages`` / ``history`` /
    ``slots`` / ``model``) so a Phase 4 implementation can swap in
    without callers having to rename. Protocol signature checking is
    structural in mypy; this test pins the parameter names at runtime
    so a refactor that drifts ``slots`` → ``state`` (etc.) fails CI
    immediately.
    """
    mc = inspect.signature(LegacyCompactor.microcompact)
    assert list(mc.parameters)[1:] == ["messages", "slots"]

    rx = inspect.signature(LegacyCompactor.reactive)
    assert list(rx.parameters)[1:] == ["history", "slots"]

    au = inspect.signature(LegacyCompactor.auto)
    # ``model`` is keyword-only per spec; ``inspect.Parameter.kind``
    # check enforces it.
    assert list(au.parameters)[1:] == ["history", "slots", "model"]
    assert au.parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY


def test_compactor_protocol_runtime_checkable() -> None:
    """The Protocol itself is decorated ``@runtime_checkable``.

    Without the decorator, ``isinstance(obj, Compactor)`` raises
    TypeError. The Phase 1 contract is that callers may use
    ``isinstance`` to detect a Compactor in mixed call sites (e.g.
    test helpers that monkeypatch a stub).
    """
    # Probe via a non-conforming object: ``isinstance`` must complete
    # (return False) rather than raise TypeError.
    assert isinstance(object(), Compactor) is False


def test_compactor_protocol_method_names() -> None:
    """The Protocol exposes exactly the three documented methods.

    Pinning the public method set protects against an accidental
    addition that would force every implementor to update.
    """
    # Filter dunders + Protocol bookkeeping; Protocol itself adds
    # ``_is_protocol`` / ``_is_runtime_protocol`` etc. We probe
    # ``__annotations__`` of the methods declared in the source.
    declared = {
        name for name in vars(Compactor)
        if not name.startswith("_")
        and callable(vars(Compactor)[name])
    }
    assert declared == {"microcompact", "reactive", "auto"}


def test_legacy_compactor_return_types_via_annotations() -> None:
    """Return annotations match Protocol shape (string-form preserved).

    ``from __future__ import annotations`` keeps annotations as
    strings; we resolve them via ``get_type_hints`` so a typo in the
    return type would surface as a NameError here.
    """
    # Resolve hints — will raise NameError if any forward ref is bad.
    for name in ("microcompact", "reactive", "auto"):
        hints = get_type_hints(getattr(LegacyCompactor, name))
        assert "return" in hints, f"{name} must declare a return type"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
