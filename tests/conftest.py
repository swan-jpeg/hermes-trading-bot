"""Shared test fixtures.

The pipeline's default closes_provider fetches live prices via yfinance.
Tests must stay offline and deterministic, so an autouse fixture replaces
the default with a deterministic stub for every test.
"""
from __future__ import annotations

import numpy as np
import pytest


def _stub_closes(entity: str) -> np.ndarray:
    """Deterministic offline price series for any entity (no network)."""
    rng = np.random.default_rng(abs(hash(entity)) % 2**31)
    return np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100


@pytest.fixture(autouse=True)
def _offline_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the live price fetcher with a deterministic stub."""
    import hermes_bot.market.indicators_link as il
    monkeypatch.setattr(il, "fetch_closes", _stub_closes)
