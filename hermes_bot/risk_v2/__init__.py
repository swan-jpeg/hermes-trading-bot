"""LAAG 6: Risk Engine v2 — adaptieve exposure (REPAIR + REDESIGN).

De oude RiskEngine was in de praktijk een statische `max_asset_weight=0.10`-cap
omdat drawdown, regime en Monte Carlo crash-detectie dood waren.

v2 repareert die mechanismen en bouwt een transparante, adaptieve
exposure-functie:

    final_exposure =
        base_exposure
        × volatility_multiplier
        × drawdown_multiplier
        × correlation_multiplier
        × tail_risk_multiplier
        × regime_multiplier
        × certainty_multiplier

    final_exposure = clamp(final_exposure, min_exposure, max_exposure)

Elke multiplier is afzonderlijk zichtbaar in de logs, zodat achteraf exact te
zien is waarom de engine risico verlaagde. Hysteresis/smoothing voorkomt
whipsaw; recovery-logica bouwt exposure geleidelijk weer op na een crash.

De oude RiskEngine blijft intact (baseline). Deze module is een aparte v2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.schemas import MonteCarloResult, RLRawDecision


# ---------------------------------------------------------------------------
# REGIME-DETECTIE — eenvoudig, uitlegbaar, alleen historische info
# ---------------------------------------------------------------------------
def detect_regime(
    drawdown: float,
    vol: float,
    vol_baseline: float,
    crash_prob: float,
    recovery_signal: float,
    prev_regime: str,
) -> str:
    """Bepaal het regime o.b.v. alleen nu-beschikbare informatie.

    Regimes: normal, elevated, stressed, crash, recovery.
    - crash: drawdown < -15% OF crash_prob > 0.30
    - stressed: drawdown < -8% OF vol > 2.0× baseline
    - elevated: vol > 1.5× baseline OF crash_prob > 0.15
    - recovery: drawdown verbetert (> -5%) na een crash/stressed periode
    - normal: anders
    """
    if drawdown < -0.15 or crash_prob > 0.30:
        return "crash"
    if drawdown < -0.08 or vol > 2.0 * vol_baseline:
        return "stressed"
    if vol > 1.5 * vol_baseline or crash_prob > 0.15:
        return "elevated"
    if prev_regime in ("crash", "stressed") and drawdown > -0.05 and recovery_signal > 0:
        return "recovery"
    return "normal"


# ---------------------------------------------------------------------------
# RISK ENGINE V2
# ---------------------------------------------------------------------------
@dataclass
class MultiplierBreakdown:
    """Alle multipliers + final exposure — voor logging en reconstructie."""

    base_exposure: float
    vol_multiplier: float
    drawdown_multiplier: float
    correlation_multiplier: float
    tail_multiplier: float
    regime_multiplier: float
    certainty_multiplier: float
    final_exposure: float
    regime: str
    vol: float
    drawdown: float
    correlation: float
    var_95: float
    crash_prob: float
    reason: str


class RiskEngineV2:
    """Adaptieve risico-engine met transparante multipliers."""

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("risk", {})
        self.mc = MonteCarloEngineV2(seed=self.cfg.get("seed", 42), n_paths=2000)

        # Exposure-parameters.
        self.base_exposure = self.cfg.get("base_exposure", 0.90)
        self.min_exposure = self.cfg.get("min_exposure", 0.0)
        self.max_exposure = self.cfg.get("max_exposure", 1.0)

        # Vol-targeting.
        self.vol_target = self.cfg.get("vol_target", 0.125)
        self.vol_lookback = self.cfg.get("vol_lookback", 20)

        # Drawdown.
        self.max_drawdown = self.cfg.get("max_portfolio_drawdown", -0.08)

        # Correlation.
        self.corr_lookback = self.cfg.get("corr_lookback", 20)
        self.corr_threshold = self.cfg.get("corr_threshold", 0.6)

        # Tail risk.
        self.max_var_95 = self.cfg.get("max_var_95", -0.05)
        self.max_crash_prob = self.cfg.get("max_crash_prob", 0.10)

        # Hysteresis / smoothing.
        self.max_daily_change = self.cfg.get("max_daily_change", 0.25)
        self.min_regime_hold = self.cfg.get("min_regime_hold", 5)

        # Recovery.
        self.recovery_lookback = self.cfg.get("recovery_lookback", 10)

        # Ablation-toggles (voor experiment B): zet een multiplier uit.
        self._vol_off = self.cfg.get("_vol_off", False)
        self._dd_off = self.cfg.get("_dd_off", False)
        self._corr_off = self.cfg.get("_corr_off", False)
        self._mc_off = self.cfg.get("_mc_off", False)
        self._regime_off = self.cfg.get("_regime_off", False)
        self._certainty_off = self.cfg.get("_certainty_off", False)

        # State.
        self.peak_equity = 0.0
        self.prev_exposure = 0.0
        self.prev_regime = "normal"
        self.regime_days = 0
        self.hist_returns: list[float] = []
        self.hist_prices: list[float] = []

    # --- Drawdown (echt, geen look-ahead) ---
    def update_drawdown(self, equity: float) -> float:
        """Update peak equity en bereken drawdown. Alleen historische info."""
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity <= 0:
            return 0.0
        return equity / self.peak_equity - 1.0

    # --- Volatiliteit (alleen historische returns) ---
    def _volatility(self) -> float:
        if len(self.hist_returns) < 2:
            return 0.0
        rets = np.asarray(self.hist_returns[-self.vol_lookback :])
        if rets.std() == 0:
            return 0.0
        return float(rets.std() * np.sqrt(252))

    # --- Correlatie (alleen historische returns) ---
    def _correlation(self) -> float:
        """Autocorrelatie van returns als proxy voor markt-correlatie.

        In een single-asset test is er geen cross-asset correlatie; we gebruiken
        de autocorrelatie (momentum/mean-reversion) als proxy. In multi-asset
        zou dit de gemiddelde pairwise correlatie zijn.
        """
        if len(self.hist_returns) < self.corr_lookback + 1:
            return 0.0
        rets = np.asarray(self.hist_returns[-(self.corr_lookback + 1) :])
        x, y = rets[:-1], rets[1:]
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    # --- Recovery-signaal ---
    def _recovery_signal(self) -> float:
        """Positief als recente returns gemiddeld positief zijn (trend herstel)."""
        if len(self.hist_returns) < self.recovery_lookback:
            return 0.0
        recent = np.asarray(self.hist_returns[-self.recovery_lookback :])
        return float(np.mean(recent))

    # --- Multipliers ---
    def _vol_multiplier(self, vol: float) -> float:
        if self._vol_off or vol <= 0:
            return 1.0
        return float(np.clip(self.vol_target / vol, 0.1, 1.0))

    def _drawdown_multiplier(self, drawdown: float) -> float:
        if self._dd_off or drawdown >= 0:
            return 1.0
        # Hoe dieper de drawdown, hoe lager de exposure.
        # ratio = drawdown / max_drawdown (0..1+). ratio=0 -> 1.0, ratio>=1 -> 0.1.
        ratio = drawdown / self.max_drawdown
        return float(np.clip(1.0 - 0.9 * ratio, 0.1, 1.0))

    def _correlation_multiplier(self, corr: float) -> float:
        if self._corr_off or corr <= 0:
            return 1.0
        return float(np.clip(1.0 - (corr - 0.3) / (self.corr_threshold - 0.3) * 0.5, 0.5, 1.0))

    def _tail_multiplier(self, var_95: float, crash_prob: float) -> float:
        if self._mc_off:
            return 1.0
        m = 1.0
        if var_95 < self.max_var_95:
            m *= max(0.1, self.max_var_95 / var_95)
        if crash_prob > self.max_crash_prob:
            m *= 0.5
        return float(np.clip(m, 0.1, 1.0))

    def _regime_multiplier(self, regime: str) -> float:
        if self._regime_off:
            return 1.0
        return {
            "normal": 1.0, "elevated": 0.75, "stressed": 0.5,
            "crash": 0.25, "recovery": 0.6,
        }.get(regime, 1.0)

    def _certainty_multiplier(self, certainty: float) -> float:
        if self._certainty_off:
            return 1.0
        return float(0.5 + 0.5 * certainty)

    # --- Smoothing / hysteresis ---
    def _smooth(self, target: float) -> float:
        """Beperk dagelijkse verandering (voorkomt whipsaw)."""
        delta = target - self.prev_exposure
        max_delta = self.max_daily_change
        if abs(delta) > max_delta:
            target = self.prev_exposure + np.sign(delta) * max_delta
        return float(np.clip(target, self.min_exposure, self.max_exposure))

    # --- Main entry point ---
    def approve(
        self,
        decision: RLRawDecision,
        portfolio: PortfolioState,
        equity: float,
        mc: MonteCarloResult | None = None,
    ) -> tuple[float, MultiplierBreakdown]:
        """Bereken de goedgekeurde exposure met volledige multiplier-breakdown."""
        if decision.action.value == "sell" and decision.exit_reason.value != "none":
            self.prev_exposure = 0.0
            return 0.0, MultiplierBreakdown(
                base_exposure=0.0, vol_multiplier=1.0, drawdown_multiplier=1.0,
                correlation_multiplier=1.0, tail_multiplier=1.0,
                regime_multiplier=1.0, certainty_multiplier=1.0,
                final_exposure=0.0, regime=self.prev_regime, vol=0.0,
                drawdown=0.0, correlation=0.0, var_95=0.0, crash_prob=0.0,
                reason=f"exit: {decision.exit_reason.value}",
            )

        drawdown = self.update_drawdown(equity)
        vol = self._volatility()
        corr = self._correlation()
        recovery = self._recovery_signal()
        var_95 = mc.var_95 if mc else 0.0
        crash_prob = mc.crash_probability if mc else 0.0

        vol_baseline = max(0.05, self.vol_target)

        # Regime met hysteresis (min hold-tijd).
        regime = detect_regime(drawdown, vol, vol_baseline, crash_prob, recovery,
                               self.prev_regime)
        if regime == self.prev_regime:
            self.regime_days += 1
        else:
            if self.regime_days < self.min_regime_hold and self.prev_regime != "normal":
                regime = self.prev_regime
            else:
                self.regime_days = 0
        self.prev_regime = regime

        base = self.base_exposure
        vm = self._vol_multiplier(vol)
        dm = self._drawdown_multiplier(drawdown)
        cm = self._correlation_multiplier(corr)
        tm = self._tail_multiplier(var_95, crash_prob)
        rm = self._regime_multiplier(regime)
        ctm = self._certainty_multiplier(decision.zekerheid)

        raw = base * vm * dm * cm * tm * rm * ctm
        final = self._smooth(raw)

        reasons = []
        if vm < 0.99:
            reasons.append(f"vol={vol:.3f}")
        if dm < 0.99:
            reasons.append(f"dd={drawdown:.3f}")
        if cm < 0.99:
            reasons.append(f"corr={corr:.2f}")
        if tm < 0.99:
            reasons.append(f"var={var_95:.3f}/crash={crash_prob:.2f}")
        if rm < 0.99:
            reasons.append(f"regime={regime}")
        if ctm < 0.99:
            reasons.append(f"certainty={decision.zekerheid:.2f}")
        reason = ", ".join(reasons) if reasons else "normal"

        self.prev_exposure = final
        return final, MultiplierBreakdown(
            base_exposure=round(base, 4), vol_multiplier=round(vm, 4),
            drawdown_multiplier=round(dm, 4), correlation_multiplier=round(cm, 4),
            tail_multiplier=round(tm, 4), regime_multiplier=round(rm, 4),
            certainty_multiplier=round(ctm, 4), final_exposure=round(final, 4),
            regime=regime, vol=round(vol, 4), drawdown=round(drawdown, 4),
            correlation=round(corr, 4), var_95=round(var_95, 4),
            crash_prob=round(crash_prob, 4), reason=reason,
        )

    def record_return(self, ret: float) -> None:
        """Registreer een dagrendement voor vol/corr/recovery-berekening."""
        self.hist_returns.append(ret)
        if len(self.hist_returns) > 500:
            self.hist_returns = self.hist_returns[-500:]
