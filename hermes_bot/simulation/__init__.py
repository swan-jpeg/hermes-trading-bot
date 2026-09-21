"""LAAG 6: Monte Carlo simulatie — probabilistische risicoroute. [[08]]"""
from __future__ import annotations

import numpy as np

from hermes_bot.schemas import MonteCarloResult


class MonteCarloEngine:
    def __init__(self, seed: int = 42, n_paths: int = 10_000) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_paths = n_paths

    def simulate(
        self,
        returns: np.ndarray,
        horizon: int,
        corr: np.ndarray | None = None,
    ) -> MonteCarloResult:
        """TODO: bootstrap/blokbootstrap van rendementen + correlatie.
        Bepaal percentielen, VaR95, ES95, drawdown-distributie, crashprob.
        """
        raise NotImplementedError

    def stress_scenarios(self, current_portfolio: dict, scenarios: dict) -> dict:
        """Test huidige portefeuille tegen crash-scenario's (2008/2020/2022)."""
        raise NotImplementedError
