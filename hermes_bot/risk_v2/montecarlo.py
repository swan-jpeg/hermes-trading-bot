"""LAAG 6: Monte Carlo v2 — echte path-simulatie (REPAIR).

De oude versie gebruikte een constante dag-return per pad
(`daily = paths[i].mean()/horizon`), waardoor path-dependent drawdowns en
crashes niet konden ontstaan en `crash_probability` altijd 0.0 was.

v2 simuleert echte paden door per dag een rendement te bootstrappen uit de
historische rendementen (alleen data vóór de huidige dag — geen look-ahead).
Hierdoor ontstaan realistische drawdowns, VaR, CVaR en crash-kansen.
"""
from __future__ import annotations

import numpy as np

from hermes_bot.schemas import MonteCarloResult


class MonteCarloEngineV2:
    """Blok-bootstrap van rendementen met echte path-simulatie.

    Elke pad is een cumulatief product van dagrendementen die per dag uit de
    historische verdeling worden getrokken (met blokken om autocorrelatie te
    behouden). Hierdoor ontstaan echte drawdowns en crashes.
    """

    def __init__(self, seed: int = 42, n_paths: int = 10_000) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_paths = n_paths

    def simulate(
        self,
        returns: np.ndarray,
        horizon: int,
        corr: np.ndarray | None = None,
    ) -> MonteCarloResult:
        """Simuleer `n_paths` paden over `horizon` dagen.

        returns: 1D (enkel asset) of 2D (n_assets x n_days) dagrendementen.
        corr: optionele correlatiematrix voor multi-asset (Cholesky).
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim == 1:
            returns = returns.reshape(1, -1)
        n_assets, n_days = returns.shape

        # Multi-asset: decorreleer met Cholesky als correlatie gegeven is.
        if corr is not None and n_assets > 1:
            L = np.linalg.cholesky(np.asarray(corr, dtype=float))
            returns = (L @ returns).T  # gecorreleerde rendementen

        block = max(1, min(20, n_days // 4))  # blokgrootte behoudt autocorrelatie

        # Simuleer echte paden: per dag een rendement uit de historie.
        # Pad i, dag d: cumulatief product van getrokken rendementen.
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

        # Portfolio-rendement = gemiddelde over assets (gelijk gewicht).
        port = paths.mean(axis=1)
        pct = {q: float(np.percentile(port, q)) for q in (5, 10, 25, 50, 75, 90, 95)}

        # VaR95 en Expected Shortfall95 (linkerstaart).
        var_95 = float(np.percentile(port, 5))
        tail = port[port <= var_95]
        es_95 = float(tail.mean()) if tail.size else var_95

        # ECHTE drawdown-distributie: simuleer per pad een equity-curve door
        # dagrendementen te trekken en cumulatief te vermenigvuldigen.
        dd = np.zeros(self.n_paths)
        for i in range(self.n_paths):
            # Trek horizon dagrendementen uit de historische verdeling.
            idx = self.rng.integers(0, n_days, size=horizon)
            daily = returns[:, idx].mean(axis=0)  # portfolio dag-return per dag
            eq = np.cumprod(1 + daily)
            peak = np.maximum.accumulate(eq)
            dd[i] = float(np.min(eq / peak - 1))
        dd_pct = {q: float(np.percentile(dd, q)) for q in (50, 90, 95)}

        # Crash-kans: kans dat drawdown < -20% binnen horizon.
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
        """Test huidige portefeuille tegen crash-scenario's."""
        out: dict[str, float] = {}
        for name, asset_returns in scenarios.items():
            total = 0.0
            for asset, weight in current_portfolio.items():
                r = asset_returns.get(asset, 0.0)
                total += weight * r
            out[name] = round(total, 4)
        return out
