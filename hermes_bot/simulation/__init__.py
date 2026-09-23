"""LAAG 6: Monte Carlo simulatie — probabilistische risicoroute. [[08]]

Implementatie: blok-bootstrap van historische rendementen (behoudt
autocorrelatie en crashes) + VaR/Expected Shortfall/drawdown-distributie.
"""
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
        """Block bootstrap of returns over `horizon` days.

        returns: 1D (enkel asset) of 2D (n_assets x n_days) dagrendementen.
        corr: optionele correlatiematrix voor multi-asset (Cholesky).
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)
        n_assets, n_days = returns.shape

        # Multi-asset: decorrelate with Cholesky if correlation is given.
        if corr is not None and n_assets > 1:
            L = np.linalg.cholesky(np.asarray(corr, dtype=float))
            returns = (L @ returns).T  # gecorreleerde rendementen

        # Block bootstrap: pick random start days, loop horizon days.
        block = max(1, min(20, n_days // 4))  # blokgrootte behoudt autocorrelatie
        paths = np.zeros((self.n_paths, n_assets))
        for i in range(self.n_paths):
            cum = np.zeros(n_assets)
            day = 0
            while day < horizon:
                start = self.rng.integers(0, n_days - block)
                seg = returns[:, start : start + block]
                take = min(block, horizon - day)
                cum += seg[:, :take].sum(axis=1)
                day += take
            paths[i] = cum

        # Portfolio return = average over assets (equal weight).
        port = paths.mean(axis=1)
        pct = {q: float(np.percentile(port, q)) for q in (5, 10, 25, 50, 75, 90, 95)}

        # VaR95 and Expected Shortfall95 (left tail).
        var_95 = float(np.percentile(port, 5))
        tail = port[port <= var_95]
        es_95 = float(tail.mean()) if tail.size else var_95

        # Drawdown-distributie: max cumulatief verlies per pad.
        # Per path: average daily return over assets -> equity curve over horizon.
        dd = np.zeros(self.n_paths)
        for i in range(self.n_paths):
            daily = paths[i].mean() / horizon  # gelijkmatig verdeeld over horizon
            eq = np.cumprod(1 + np.full(horizon, daily))
            peak = np.maximum.accumulate(eq)
            dd[i] = float(np.min(eq / peak - 1))
        dd_pct = {q: float(np.percentile(dd, q)) for q in (50, 90, 95)}

        # Crash probability: chance that drawdown < -20% within the horizon.
        crash_probability = float(np.mean(dd < -0.20))

        return MonteCarloResult(
            n_paths=self.n_paths,
            horizon=horizon,
            percentiles=pct,
            var_95=var_95,
            expected_shortfall_95=es_95,
            max_drawdown_distribution=dd_pct,
            crash_probability=crash_probability,
        )

    def stress_scenarios(self, current_portfolio: dict, scenarios: dict) -> dict:
        """Test the current portfolio against crash scenarios.

        scenarios: {naam: {asset: rendement}} bijv. {"2008": {"AAPL": -0.5, ...}}.
        Retourneert per scenario het portfolio-rendement.
        """
        out: dict[str, float] = {}
        for name, asset_returns in scenarios.items():
            total = 0.0
            for asset, weight in current_portfolio.items():
                r = asset_returns.get(asset, 0.0)
                total += weight * r
            out[name] = round(total, 4)
        return out
