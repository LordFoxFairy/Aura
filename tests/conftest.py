"""Shared pytest fixtures for Aura tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_aura_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove AURA_CONFIG from the environment for every test.

    Without this, a CI runner or local shell with AURA_CONFIG set would leak
    into any test that calls load_config (directly or via Agent.__init__).
    """
    monkeypatch.delenv("AURA_CONFIG", raising=False)
