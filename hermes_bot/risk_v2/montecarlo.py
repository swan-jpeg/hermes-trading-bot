"""LEVEL 6: Monte Carlo — the probabilistic risk route ([[08]]).

This is the ONLY Monte Carlo implementation in the bot (single source of truth).
The legacy `hermes_bot/simulation/` module is now a thin shim re-exporting this
class, so there is no second, divergent engine any more.

What the engine does (in order of value, see note [[08]]):

1. **Portfolio returns (multi-asset)** — thousands of paths of the full
   portfolio -> distribution -> VaR95 / Expected Shortfall95.
2. **Price paths** — four models:
   - `bootstrap` (default): block bootstrap of historical daily returns,
     keeps autocorrelation AND the real cross-correlation/crashes from history;
   - `gbm`: geometric Brownian motion (normal, mu/sigma from history);
   - `jump_diffusion`: base path + Poisson jumps (fat tail of real crashes),
     jump intensity/size estimated automatically from history;
   - `regime`: 2-state Markov volatility regime (calm/stressed) with
     empirically estimated transition probabilities.
3. **Tail dependency** — Cholesky correlation OR a copula (`gaussian` or `t`)
   on the empirical marginals. The t-copula adds tail dependence (assets
   crash together), which Cholesky cannot.
4. **Forced crash scenarios** — `stress_scenarios()` + `DEFAULT_SCENARIOS`
   (2008, correlation spike, flash crash, ...): measure the impact on the
   CURRENT portfolio. The result lands in `MonteCarloResult.scenario_table`.
5. **Backtest uncertainty** — `bootstrap_ci()`: confidence interval of
   return/Sharpe/max-drawdown, so a backtest does not rest on one history.

Correctness vs. the old versions:
- Quantiles, VaR, ES, drawdown AND crash probability come from the SAME
  simulated paths (the old v2 re-drew the drawdown i.i.d.; the old v1 used a
  constant daily return per path -> drawdown 0.0 and crash probability 0.0).
- No look-ahead: only the supplied history is sampled.
- numpy only: `scipy` is an optional extra here (`uv sync --extra risk`).
"""
from __future__ import annotations

import numpy as np

from hermes_bot.schemas import MonteCarloResult

__all__ = [
    "DEFAULT_SCENARIOS",
    "DD_PERCENTILES",
    "MODELS",
    "PERCENTILES",
    "MonteCarloEngineV2",
]

PERCENTILES: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)
DD_PERCENTILES: tuple[int, ...] = (50, 90, 95)
MODELS: tuple[str, ...] = ("bootstrap", "gbm", "jump_diffusion", "regime")
COPULAS: tuple[str, ...] = ("gaussian", "t")

# Crash scenarios (point 3 of [[08]]): force extreme events and measure
# the impact on the current portfolio. "*" is the fallback for any instrument;
# a key per asset class (STOCK/ETF/BOND/CASH/OPTION) overrides that one.
DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "mild_correction": {"*": -0.08},
    "bear_market": {"*": -0.20, "BOND": 0.04, "CASH": 0.0},
    "correlation_spike": {"*": -0.25, "BOND": -0.03},
    "rate_shock": {"*": -0.12, "BOND": -0.08},
    "flash_crash": {"*": -0.10},
    "2008_crisis": {"*": -0.40, "BOND": 0.05, "CASH": 0.0},
}

_TRADING_DAYS = 252.0


class MonteCarloEngineV2:
    """Monte Carlo engine: paths, starting risk and crash scenarios.

    seed: reproducibility (same seed + data -> identical result).
    n_paths: number of simulated paths.
    crash_threshold: drawdown threshold for `crash_probability` (default -20%).
    model: default path model (`bootstrap` | `gbm` | `jump_diffusion` | `regime`).
    copula: default dependency model (`None` | `gaussian` | `t`).
    copula_df: degrees of freedom of the t-copula (lower = fatter tails).
    block: block size of the bootstrap (keeps autocorrelation).
    """

    def __init__(
        self,
        seed: int = 42,
        n_paths: int = 10_000,
        crash_threshold: float = -0.20,
        model: str = "bootstrap",
        copula: str | None = None,
        copula_df: float = 5.0,
        block: int = 20,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
        if copula is not None and copula not in COPULAS:
            raise ValueError(f"unknown copula {copula!r}; choose from {COPULAS}")
        self.rng = np.random.default_rng(seed)
        self.n_paths = int(n_paths)
        self.crash_threshold = float(crash_threshold)
        self.model = model
        self.copula = copula
        self.copula_df = float(copula_df)
        self.block = int(block)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def simulate(
        self,
        returns: np.ndarray,
        horizon: int,
        corr: np.ndarray | None = None,
        *,
        model: str | None = None,
        jumps: dict | None = None,
        copula: str | None = None,
        copula_df: float | None = None,
        shock: dict | float | None = None,
        shock_scale: float = 0.005,
        portfolio: dict[str, float] | None = None,
        scenarios: dict[str, dict[str, float]] | None = None,
    ) -> MonteCarloResult:
        """Simulate `n_paths` paths over `horizon` days.

        returns: 1D (single asset) or 2D (n_assets x n_days) daily returns.
        corr: optional correlation matrix (Cholesky) for multi-asset.
        model/jumps/copula: override the engine defaults for this run.
        shock: impact-agent shock applied as extra daily drift; either a dict
            {"direction": 0..1, "magnitude": 0..1} (scalars or per-asset
            arrays) or a magnitude scalar.
        portfolio/scenarios: when given, the crash-scenario stress table of that
            portfolio is included in the result (`scenario_table`).
        """
        hist = self._as_2d(returns)
        n_assets = hist.shape[0]
        horizon = max(1, int(horizon))
        model = model or self.model
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
        copula = self.copula if copula is None else copula
        copula_df = self.copula_df if copula_df is None else float(copula_df)
        if copula is not None and copula not in COPULAS:
            raise ValueError(f"unknown copula {copula!r}; choose from {COPULAS}")

        drift = self._shock_drift(shock, n_assets, shock_scale)
        paths = self._sample(hist, horizon, corr, model, jumps, copula, copula_df)
        if np.any(drift != 0.0):
            paths = paths + drift[None, None, :]

        table = self.stress_scenarios(portfolio, scenarios) if portfolio else {}
        return self._summarise(paths, horizon, model, table)

    def simulate_with_shock(
        self,
        returns: np.ndarray,
        horizon: int,
        direction: float | np.ndarray = 0.5,
        magnitude: float | np.ndarray = 0.0,
        *,
        corr: np.ndarray | None = None,
        model: str | None = None,
        jumps: dict | None = None,
        copula: str | None = None,
        copula_df: float | None = None,
        shock_scale: float = 0.005,
        portfolio: dict[str, float] | None = None,
        scenarios: dict[str, dict[str, float]] | None = None,
    ) -> MonteCarloResult:
        """Simulate paths with an impact-agent shock applied as drift.

        The impact agent determines which instruments are affected and how
        strongly (direction 0..1 -> -1..1, magnitude 0..1). This immediately
        feeds the Monte Carlo so the tail risk reflects the NEW event, not just
        past history. The shock shifts the return distribution:

            drift = (direction - 0.5) * 2 * magnitude * shock_scale

        `shock_scale` (~0.5% per day) keeps the shock comparable to daily vol
        while still bending the median path toward the impact direction.
        Works for single AND multi-asset input (direction/magnitude may be a
        scalar or one value per asset).
        """
        shock = {
            "direction": np.asarray(direction, dtype=float),
            "magnitude": np.asarray(magnitude, dtype=float),
        }
        return self.simulate(
            returns, horizon, corr, model=model, jumps=jumps, copula=copula,
            copula_df=copula_df, shock=shock, shock_scale=shock_scale,
            portfolio=portfolio, scenarios=scenarios,
        )

    def stress_scenarios(
        self,
        current_portfolio: dict[str, float] | None,
        scenarios: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Test the current portfolio against forced crash scenarios.

        scenarios: {name: {asset: return}}; the "*" key is the fallback for every
        instrument not named explicitly. Weights are fractions of the portfolio
        (cash is simply not listed and is therefore unaffected); if the weights
        sum above 1 (percentages or leverage) they are normalised first.
        Returns the portfolio return per scenario.
        """
        scen = scenarios if scenarios is not None else DEFAULT_SCENARIOS
        weights = {str(k): float(v) for k, v in (current_portfolio or {}).items()}
        gross = sum(abs(w) for w in weights.values())
        if gross > 1.0 + 1e-9:  # percentages/leverage -> eerst normaliseren
            weights = {k: w / gross for k, w in weights.items()}

        out: dict[str, float] = {}
        for name, asset_returns in scen.items():
            lookup = {str(k).upper(): float(v) for k, v in asset_returns.items()}
            fallback = lookup.get("*", 0.0)
            total = 0.0
            for asset, weight in weights.items():
                total += weight * lookup.get(str(asset).upper(), fallback)
            out[name] = round(total, 4)
        return out

    def worst_scenario(self, table: dict[str, float]) -> tuple[str, float]:
        """Name + value of the heaviest crash scenario in a stress table."""
        if not table:
            return "", 0.0
        name = min(table, key=lambda k: table[k])
        return name, float(table[name])

    def bootstrap_ci(
        self,
        returns: np.ndarray,
        alpha: float = 0.05,
        n_resamples: int = 2_000,
    ) -> dict[str, dict[str, float]]:
        """Confidence interval of strategy stats via block-bootstrap.

        Point 4 of [[08]]: resample the history in blocks (so autocorrelation
        and crashes stay in) and recompute mean/Sharpe/max-drawdown. That shows
        whether a backtest result rests on one lucky history.
        """
        r = self._as_1d(returns)
        if r.size < 2:
            zero = {"point": 0.0, "low": 0.0, "high": 0.0}
            return {"mean": dict(zero), "sharpe": dict(zero), "max_drawdown": dict(zero)}
        n_resamples = max(2, int(n_resamples))
        idx = self._block_indices(r.size, r.size, n_paths=n_resamples)
        samples = r[idx]  # (n_resamples, n)

        stds = samples.std(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpes = np.where(stds > 0, samples.mean(axis=1) / stds * np.sqrt(_TRADING_DAYS), 0.0)
        equity = np.cumprod(1.0 + samples, axis=1)
        peaks = np.maximum.accumulate(
            np.concatenate([np.ones((samples.shape[0], 1)), equity], axis=1), axis=1)[:, 1:]
        dds = np.min(equity / peaks - 1.0, axis=1)

        lo, hi = (alpha / 2) * 100, (1 - alpha / 2) * 100
        out: dict[str, dict[str, float]] = {}
        for name, point, dist in (
            ("mean", float(r.mean()), samples.mean(axis=1)),
            ("sharpe", _sharpe_of(r), sharpes),
            ("max_drawdown", _min_drawdown_of(r), dds),
        ):
            out[name] = {
                "point": round(point, 6),
                "low": round(float(np.percentile(dist, lo)), 6),
                "high": round(float(np.percentile(dist, hi)), 6),
            }
        return out

    # ------------------------------------------------------------------ #
    # Path generation
    # ------------------------------------------------------------------ #
    def _sample(
        self,
        hist: np.ndarray,
        horizon: int,
        corr: np.ndarray | None,
        model: str,
        jumps: dict | None,
        copula: str | None,
        copula_df: float,
    ) -> np.ndarray:
        """Return (n_paths, horizon, n_assets) daily returns."""
        mu = hist.mean(axis=1)
        sd = hist.std(axis=1)
        sd = np.where(sd > 0, sd, 1e-9)

        if copula is not None:
            # Copula mode: draw from the empirical marginals per day with an
            # explicit dependency structure (tail dependence), but without
            # serial autocorrelation.
            base = self._sample_copula(hist, horizon, corr, copula, copula_df)
        elif model == "gbm":
            z = self._correlate(
                self.rng.standard_normal((self.n_paths, horizon, hist.shape[0])), corr, hist)
            base = mu[None, None, :] + sd[None, None, :] * z
        elif model == "regime":
            base = self._sample_regime(hist, horizon, corr, mu, sd)
        else:  # bootstrap (and the base of jump_diffusion)
            base = self._sample_bootstrap(hist, horizon, corr, mu, sd)

        if model == "jump_diffusion":
            base = base + self._sample_jumps(hist, horizon, jumps)
        return base

    def _sample_bootstrap(
        self,
        hist: np.ndarray,
        horizon: int,
        corr: np.ndarray | None,
        mu: np.ndarray,
        sd: np.ndarray,
    ) -> np.ndarray:
        """Blok-bootstrap: behoudt autocorrelatie én historische kruiscorrelatie."""
        n_assets, n_days = hist.shape
        idx = self._block_indices(n_days, horizon)
        draw = hist[:, idx].transpose(1, 2, 0)  # (n_paths, horizon, n_assets)
        if corr is None or n_assets < 2:
            return draw
        # Set to z-scores, enforce target correlation, scale back.
        z = (draw - mu[None, None, :]) / sd[None, None, :]
        return mu[None, None, :] + sd[None, None, :] * (z @ self._cholesky(corr).T)

    def _sample_regime(
        self,
        hist: np.ndarray,
        horizon: int,
        corr: np.ndarray | None,
        mu: np.ndarray,
        sd: np.ndarray,
    ) -> np.ndarray:
        """2-toestands Markov-volatiliteitsregime (rustig=0, gespannen=1)."""
        n_assets, n_days = hist.shape
        market = hist.mean(axis=0)
        window = max(2, min(20, n_days // 4))
        vol = _rolling_std(market, window)
        threshold = float(np.median(vol)) if vol.size else 0.0
        states = (vol > threshold).astype(int)

        sigmas = np.zeros(2)
        means = np.zeros(2)
        for k in (0, 1):
            sel = market[states == k]
            sigmas[k] = float(sel.std()) if sel.size > 1 else float(market.std() or 1e-9)
            means[k] = float(sel.mean()) if sel.size else float(market.mean())
        sigmas = np.where(sigmas > 0, sigmas, 1e-9)

        # Transition probabilities with Laplace smoothing (empirical from the history).
        trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        for k in (0, 1):
            same = int(np.sum((states[:-1] == k) & (states[1:] == k)))
            total = int(np.sum(states[:-1] == k))
            trans[k, 0] = (same + 1.0) / (total + 2.0)
            trans[k, 1] = 1.0 - trans[k, 0]

        chain = np.zeros((self.n_paths, horizon), dtype=int)
        prev = self.rng.integers(0, 2, size=self.n_paths)
        chain[:, 0] = prev
        for t in range(1, horizon):
            u = self.rng.random(self.n_paths)
            stay = u < trans[prev, prev]
            prev = np.where(stay, prev, 1 - prev)
            chain[:, t] = prev

        scale = (sigmas / sigmas[0])[chain]  # (n_paths, horizon)
        drift_shift = (means[chain] - means[0])[:, :, None]
        z = self._correlate(
            self.rng.standard_normal((self.n_paths, horizon, n_assets)), corr, hist)
        return mu[None, None, :] + drift_shift + sd[None, None, :] * scale[:, :, None] * z

    def _sample_copula(
        self,
        hist: np.ndarray,
        horizon: int,
        corr: np.ndarray | None,
        kind: str,
        copula_df: float,
    ) -> np.ndarray:
        """Copula on empirical marginals (Iman-Conover).

        `gaussian` = normal copula (no tail dependence), `t` = t-copula
        (assets crash together). The marginals stay exactly those of history.
        """
        n_assets, n_days = hist.shape
        corr_m = self._corr_matrix(corr, hist)
        g = self.rng.standard_normal((self.n_paths, horizon, n_assets))
        g = g @ self._cholesky(corr_m).T
        if kind == "t":
            chi2 = self.rng.chisquare(copula_df, size=(self.n_paths, horizon, 1))
            g = g * np.sqrt(copula_df / chi2)
        u = _rank_uniform(g)
        idx = np.clip((u * n_days).astype(int), 0, n_days - 1)
        sorted_hist = np.sort(hist, axis=1)  # (n_assets, n_days)
        out = np.empty((self.n_paths, horizon, n_assets))
        for a in range(n_assets):
            out[:, :, a] = sorted_hist[a][idx[:, :, a]]
        return out

    def _sample_jumps(self, hist: np.ndarray, horizon: int, jumps: dict | None) -> np.ndarray:
        """Poisson-jumps (crash-tail) above the base path."""
        n_assets, n_days = hist.shape
        if jumps is None:
            lam, mu_j, sd_j = self._jump_params(hist)
        else:
            lam = np.broadcast_to(
                np.asarray(jumps.get("intensity", 0.01), dtype=float), (n_assets,)).astype(float)
            mu_j = np.broadcast_to(
                np.asarray(jumps.get("mean", -0.05), dtype=float), (n_assets,)).astype(float)
            sd_j = np.broadcast_to(
                np.asarray(jumps.get("std", 0.03), dtype=float), (n_assets,)).astype(float)
        counts = self.rng.poisson(lam, size=(self.n_paths, horizon, n_assets))
        sizes = mu_j[None, None, :] + sd_j[None, None, :] * self.rng.standard_normal(
            (self.n_paths, horizon, n_assets))
        return counts * sizes

    def _jump_params(self, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate jump intensity/size from history (|z| > 3 = jump)."""
        n_assets, n_days = hist.shape
        mu = hist.mean(axis=1)
        sd = hist.std(axis=1)
        sd = np.where(sd > 0, sd, 1e-9)
        masks = np.abs((hist - mu[:, None]) / sd[:, None]) > 3.0
        lam = masks.sum(axis=1) / max(1, n_days)
        mu_j = np.empty(n_assets)
        sd_j = np.empty(n_assets)
        for a in range(n_assets):
            days = hist[a][masks[a]]
            mu_j[a] = float(days.mean()) if days.size else float(-3.0 * sd[a])
            sd_j[a] = float(days.std()) if days.size > 1 else float(0.5 * sd[a])
        return lam, mu_j, sd_j

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _block_indices(self, n_days: int, horizon: int, n_paths: int | None = None) -> np.ndarray:
        """(n_paths, horizon) index matrix of blocks from the history."""
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        n_days = max(1, int(n_days))
        block = max(1, min(self.block, n_days))
        max_start = max(1, n_days - block + 1)
        n_blocks = int(np.ceil(horizon / block))
        starts = self.rng.integers(0, max_start, size=(n_paths, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, -1)
        return np.clip(idx, 0, n_days - 1)[:, :horizon]

    def _correlate(self, z: np.ndarray, corr: np.ndarray | None,
                   hist: np.ndarray) -> np.ndarray:
        """Correleer onafhankelijke normals; default = historische correlatie."""
        if hist.shape[0] < 2:
            return z
        return z @ self._cholesky(self._corr_matrix(corr, hist)).T

    def _corr_matrix(self, corr: np.ndarray | None, hist: np.ndarray) -> np.ndarray:
        """Correlation matrix: provided, otherwise historical."""
        n_assets = hist.shape[0]
        if corr is None:
            if n_assets < 2 or hist.shape[1] < 2:
                return np.eye(n_assets)
            m = np.corrcoef(hist)
        else:
            m = np.asarray(corr, dtype=float)
        m = np.atleast_2d(m)
        if m.shape[0] != n_assets or m.shape[1] != n_assets:
            m = np.eye(n_assets)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    def _cholesky(self, corr: np.ndarray) -> np.ndarray:
        """Cholesky with symmetrization + eigenvalue fallback (non-PSD input)."""
        m = np.asarray(corr, dtype=float)
        n = m.shape[0]
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 1.0)
        try:
            return np.linalg.cholesky(m)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(m)
            vals = np.clip(vals, 1e-8, None)
            fixed = vecs @ np.diag(vals) @ vecs.T
            d = np.sqrt(np.diag(fixed))
            fixed = fixed / np.outer(d, d)
            np.fill_diagonal(fixed, 1.0)
            try:
                return np.linalg.cholesky(fixed)
            except np.linalg.LinAlgError:
                return np.eye(n)

    def _shock_drift(self, shock: dict | float | None, n_assets: int,
                     shock_scale: float) -> np.ndarray:
        """Impact-agent schok -> extra dagelijkse drift per asset."""
        if shock is None:
            return np.zeros(n_assets)
        if isinstance(shock, dict):
            direction = np.asarray(shock.get("direction", 0.5), dtype=float)
            magnitude = np.asarray(shock.get("magnitude", 0.0), dtype=float)
        else:
            direction = np.asarray(0.5, dtype=float)
            magnitude = np.asarray(shock, dtype=float)
        signed = (np.broadcast_to(direction, (n_assets,)) - 0.5) * 2.0
        mag = np.broadcast_to(magnitude, (n_assets,))
        return signed * mag * float(shock_scale)

    def _summarise(self, paths: np.ndarray, horizon: int, model: str,
                   scenario_table: dict[str, float]) -> MonteCarloResult:
        """VaR/ES/drawdown/crash probability from THE SAME paths as the quantiles."""
        n_paths = int(paths.shape[0])
        port_daily = paths.mean(axis=2) if paths.shape[2] > 1 else paths[:, :, 0]
        equity = np.cumprod(1.0 + port_daily, axis=1)
        # The peak starts at the start value (1.0); otherwise a first-day loss
        # would not count as drawdown.
        with_start = np.concatenate([np.ones((n_paths, 1)), equity], axis=1)
        peaks = np.maximum.accumulate(with_start, axis=1)[:, 1:]
        dd = np.min(equity / peaks - 1.0, axis=1)
        terminal = equity[:, -1] - 1.0

        pct = {q: float(np.percentile(terminal, q)) for q in PERCENTILES}
        var_95 = float(np.percentile(terminal, 5))
        tail = terminal[terminal <= var_95]
        es_95 = float(tail.mean()) if tail.size else var_95
        dd_pct = {q: float(np.percentile(dd, q)) for q in DD_PERCENTILES}
        crash_probability = float(np.mean(dd < self.crash_threshold))

        return MonteCarloResult(
            n_paths=n_paths,
            horizon=horizon,
            percentiles=pct,
            var_95=var_95,
            expected_shortfall_95=es_95,
            max_drawdown_distribution=dd_pct,
            crash_probability=crash_probability,
            model=model,
            scenario_table=scenario_table,
        )

    def _as_2d(self, returns: np.ndarray) -> np.ndarray:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"returns must be 1D or 2D, got shape {arr.shape}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _as_1d(self, returns: np.ndarray) -> np.ndarray:
        arr = self._as_2d(returns)
        return arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #
def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (small windows; no pandas/scipy needed)."""
    out = np.zeros(len(x), dtype=float)
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        out[t] = float(np.std(x[lo : t + 1]))
    return out


def _rank_uniform(g: np.ndarray) -> np.ndarray:
    """For each asset: ranges from the copula draw -> uniform (0,1) values."""
    out = np.empty_like(g, dtype=float)
    n_paths, horizon, n_assets = g.shape
    flat_size = n_paths * horizon
    for a in range(n_assets):
        flat = g[:, :, a].reshape(-1)
        order = np.argsort(flat, kind="stable")
        ranks = np.empty(flat_size, dtype=float)
        ranks[order] = np.arange(flat_size, dtype=float)
        out[:, :, a] = ((ranks + 0.5) / flat_size).reshape(n_paths, horizon)
    return out


def _sharpe_of(returns: np.ndarray) -> float:
    """Annualised Sharpe (used by `bootstrap_ci`)."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    sd = r.std()
    if r.size == 0 or sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(_TRADING_DAYS))


def _min_drawdown_of(returns: np.ndarray) -> float:
    """Worst drawdown of the equity curve of `returns` (used by `bootstrap_ci`)."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float(np.min(equity / peak - 1.0))
