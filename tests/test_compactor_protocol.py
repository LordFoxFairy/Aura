"""Phase 1 Task 11 — :class:`Compactor` Protocol conformance.

Verifies that :class:`aura.core.compact.compactor.Compactor` satisfies the
:class:`aura.core.compact.Compactor` Protocol both via
``runtime_checkable`` ``isinstance`` *and* via attribute-level shape
checks. The Protocol is async on every method; ``isinstance`` only
checks for attribute presence, so we additionally confirm each method
exists, is callable, and is async.

Originally pinned the Phase 1 free-function adapter; once that adapter
was removed the conformance contract still has to hold for the Phase 4
first-class implementation, so the assertions repoint to the concrete
class. The Protocol surface (``microcompact`` / ``reactive`` /
``auto``) is unchanged.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from aura.core.compact import Compactor
from aura.core.compact.compactor import Compactor as CompactorImpl


def test_compactor_impl_is_protocol_instance() -> None:
    """Class itself satisfies the runtime-checkable Protocol shape.

    ``Compactor`` is decorated ``@runtime_checkable`` so an
    ``isinstance(obj, Compactor)`` check confirms the three required
    method names (``microcompact`` / ``reactive`` / ``auto``) are
    present as attributes on the instance.

    A bare class instance can't be constructed without an Agent +
    config, but Protocol membership is structural, so we test against
    the class via a minimal ``object.__new__`` shell — that's enough
    for the isinstance check (which only walks ``__dict__``).
    """
    instance = object.__new__(CompactorImpl)
    assert isinstance(instance, Compactor)


def test_compactor_impl_methods_are_async() -> None:
    """All three Compactor methods are coroutines per spec §3.3."""
    for name in ("microcompact", "reactive", "auto"):
        method = getattr(CompactorImpl, name)
        assert callable(method), f"{name} must be callable"
        assert inspect.iscoroutinefunction(method), (
            f"{name} must be ``async def`` per Compactor Protocol"
        )


def test_compactor_impl_signatures_match_protocol() -> None:
    """Argument names match the Protocol declaration in ``__init__.py``.

    Spec §3.3 fixes the keyword names (``messages`` / ``history`` /
    ``slots`` / ``model``) so an implementation can swap in without
    callers having to rename. Protocol signature checking is structural
    in mypy; this test pins the parameter names at runtime so a
    refactor that drifts ``slots`` → ``state`` (etc.) fails CI
    immediately.

    Phase 4 added an optional ``trigger`` keyword (default-valued) on
    each method for unified ``compact_event`` tagging. The Protocol's
    structural check ignores trailing keyword-with-default parameters,
    so we assert the leading positional names match the Protocol and
    that ``trigger`` carries a default if present.
    """
    mc = inspect.signature(CompactorImpl.microcompact)
    assert list(mc.parameters)[1:3] == ["messages", "slots"]

    rx = inspect.signature(CompactorImpl.reactive)
    assert list(rx.parameters)[1:3] == ["history", "slots"]

    au = inspect.signature(CompactorImpl.auto)
    # ``model`` is keyword-only per spec; ``inspect.Parameter.kind``
    # check enforces it.
    auto_params = list(au.parameters)[1:]
    assert auto_params[:3] == ["history", "slots", "model"]
    assert au.parameters["model"].kind is inspect.Parameter.KEYWORD_ONLY

    # Extra ``trigger`` keyword (Phase 4) must carry a default so the
    # Protocol's three-arg call shape still type-checks.
    for name, sig in (
        ("microcompact", mc),
        ("reactive", rx),
        ("auto", au),
    ):
        if "trigger" in sig.parameters:
            assert sig.parameters["trigger"].default is not inspect.Parameter.empty, (
                f"{name}: ``trigger`` parameter must have a default value"
            )


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


def test_compactor_impl_return_types_via_annotations() -> None:
    """Return annotations match Protocol shape (string-form preserved).

    ``from __future__ import annotations`` keeps annotations as
    strings; we resolve them via ``get_type_hints`` so a typo in the
    return type would surface as a NameError here.
    """
    # Resolve hints — will raise NameError if any forward ref is bad.
    for name in ("microcompact", "reactive", "auto"):
        hints = get_type_hints(getattr(CompactorImpl, name))
        assert "return" in hints, f"{name} must declare a return type"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
