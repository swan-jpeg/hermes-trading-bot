"""Risk Engine v2.1 — Adaptive Risk Budgeting (evolution of frozen v2).

Centrale onderzoeksvraag:
> Kan v2.1 de crashbescherming van v2 behouden terwijl het upside-capture
> verbetert, onnodige defensieve exposure vermindert, en risico-exposure
> sneller herstelt na echte marktstabilisatie?

Architectuur-hypothese:
> v2 beantwoordt "hoe gevaarlijk is de omgeving?"
> v2.1 beantwoordt ook "hoeveel kans is er om risico te nemen?"

Drie-laags risico-architectuur:
- Layer A (Strategic): langzame baseline (vol, trend, drawdown, correlatie)
- Layer B (Tactical): snellere aanpassingen (vol-acceleratie, breadth, trend-break)
- Layer C (Emergency Brake): extreem snelle bescherming (extreme returns)

Recovery Engine: NORMAL -> ALERT -> DEFENSIVE -> STABILIZATION -> RECOVERY -> NORMAL

De frozen v2 (commit ffa398c) blijft intact. Deze module is v2.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.risk_v2.montecarlo import MonteCarloEngineV2
from hermes_bot.risk_v2_1.scores import (
    OpportunityScore,
    RiskEnvironmentScore,
    risk_budget_from_scores,
    state_label,
)
from hermes_bot.schemas import MonteCarloResult, RLRawDecision


# ---------------------------------------------------------------------------
# RECOVERY ENGINE — expliciete toestandsmachine
# ---------------------------------------------------------------------------
class RecoveryEngine:
    """Beheert de recovery-toestand: NORMAL -> ALERT -> DEFENSIVE ->
    STABILIZATION -> RECOVERY -> NORMAL.

    Recovery-signalen:
    - vol daalt
    - trend stabiliseert
    - drawdown stabiliseert
    - markt herwint trend-niveau
    """

    def __init__(self, config: dict) -> None:
        c = config.get("risk", {})
        self.stabilize_days = c.get("stabilize_days", 5)  # dagen in STABILIZATION
        self.recovery_lookback = c.get("recovery_lookback", 10)
        self.state = "NORMAL"
        self.state_days = 0

    def update(
        self,
        drawdown: float,
        vol: float,
        vol_baseline: float,
        recent_returns: list[float],
    ) -> str:
        """Update de recovery-toestand o.b.v. alleen historische info."""
        self.state_days += 1
        recovery_signal = float(np.mean(recent_returns[-self.recovery_lookback:])) \
            if len(recent_returns) >= self.recovery_lookback else 0.0
        vol_normalized = vol <= 1.3 * vol_baseline if vol_baseline > 0 else True
        dd_stabilized = drawdown > -0.05

        if self.state == "NORMAL":
            if drawdown < -0.08 or vol > 2.0 * vol_baseline:
                self.state = "ALERT"
                self.state_days = 0
        elif self.state == "ALERT":
            if drawdown < -0.12 or vol > 2.5 * vol_baseline:
                self.state = "DEFENSIVE"
                self.state_days = 0
            elif dd_stabilized and vol_normalized:
                self.state = "NORMAL"
                self.state_days = 0
        elif self.state == "DEFENSIVE":
            if dd_stabilized and vol_normalized:
                self.state = "STABILIZATION"
                self.state_days = 0
        elif self.state == "STABILIZATION":
            if self.state_days >= self.stabilize_days and recovery_signal > 0:
                self.state = "RECOVERY"
                self.state_days = 0
        elif self.state == "RECOVERY":
            if recovery_signal > 0.001 and vol_normalized:
                self.state = "NORMAL"
                self.state_days = 0
            elif drawdown < -0.10:
                self.state = "DEFENSIVE"
                self.state_days = 0
        return self.state


# ---------------------------------------------------------------------------
# EMERGENCY BRAKE — Layer C
# ---------------------------------------------------------------------------
class EmergencyBrake:
    """Extreem snelle bescherming tegen extreme events.

    Triggers: extreme dag-return, extreme vol-jump, gap-down.
    Test drempels: -10%, -15%, -20%, -30% (configurable).
    """

    def __init__(self, config: dict) -> None:
        c = config.get("risk", {})
        self.extreme_return_threshold = c.get("emergency_return_threshold", -0.10)
        self.vol_jump_threshold = c.get("emergency_vol_jump", 3.0)  # 3x vol
        self.brake_exposure = c.get("emergency_exposure", 0.0)
        self.brake_days = c.get("emergency_brake_days", 3)
        self.active = False
        self.remaining_days = 0

    def check(self, daily_return: float, vol: float, vol_baseline: float) -> bool:
        """Trigger de brake bij extreme events."""
        if daily_return <= self.extreme_return_threshold:
            self.active = True
            self.remaining_days = self.brake_days
            return True
        if vol_baseline > 0 and vol > self.vol_jump_threshold * vol_baseline:
            self.active = True
            self.remaining_days = self.brake_days
            return True
        return False

    def tick(self) -> float:
        """Geef de exposure-multiplier voor vandaag; deactiveer na brake_days."""
        if self.active:
            self.remaining_days -= 1
            if self.remaining_days <= 0:
                self.active = False
            return self.brake_exposure
        return 1.0


# ---------------------------------------------------------------------------
# RISK ENGINE V2.1
# ---------------------------------------------------------------------------
@dataclass
class V21Breakdown:
    """Volledige breakdown voor logging en reconstructie."""

    risk_score: float
    opp_score: float
    state: str
    risk_budget: float
    desired_exposure: float
    effective_exposure: float
    strategic_multiplier: float
    tactical_multiplier: float
    emergency_multiplier: float
    recovery_multiplier: float
    vol: float
    drawdown: float
    correlation: float
    var_95: float
    crash_prob: float
    opp_breakdown: dict
    reason: str


class RiskEngineV21:
    """Adaptieve risk-budgeting engine met twee-assige scores + 3 lagen."""

    def __init__(self, config: dict) -> None:
        self.cfg = config.get("risk", {})
        self.mc = MonteCarloEngineV2(seed=self.cfg.get("seed", 42), n_paths=1000)

        # Exposure.
        self.base_exposure = self.cfg.get("base_exposure", 0.90)
        self.min_exposure = self.cfg.get("min_exposure", 0.0)
        self.max_exposure = self.cfg.get("max_exposure", 1.0)
        self.max_daily_change = self.cfg.get("max_daily_change", 0.25)

        # Scores.
        self.risk_score_calc = RiskEnvironmentScore(config)
        self.opp_score_calc = OpportunityScore(config)

        # Lagen.
        self.recovery = RecoveryEngine(config)
        self.brake = EmergencyBrake(config)

        # State.
        self.peak_equity = 0.0
        self.prev_exposure = 0.0
        self.hist_returns: list[float] = []
        self.vol_baseline = max(0.05, self.cfg.get("vol_target", 0.125))

        # Ablation-toggles.
        self._opp_off = self.cfg.get("_opp_off", False)
        self._strategic_off = self.cfg.get("_strategic_off", False)
        self._tactical_off = self.cfg.get("_tactical_off", False)
        self._emergency_off = self.cfg.get("_emergency_off", False)
        self._recovery_off = self.cfg.get("_recovery_off", False)

    # --- Metrics (alleen historische info) ---
    def update_drawdown(self, equity: float) -> float:
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity <= 0:
            return 0.0
        return equity / self.peak_equity - 1.0

    def _volatility(self) -> float:
        if len(self.hist_returns) < 2:
            return 0.0
        rets = np.asarray(self.hist_returns[-20:])
        if rets.std() == 0:
            return 0.0
        return float(rets.std() * np.sqrt(252))

    def _correlation(self) -> float:
        if len(self.hist_returns) < 21:
            return 0.0
        rets = np.asarray(self.hist_returns[-21:])
        x, y = rets[:-1], rets[1:]
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _vol_acceleration(self) -> float:
        """Vol-acceleratie: recente vol vs langere-termijn vol."""
        if len(self.hist_returns) < 40:
            return 0.0
        short = np.asarray(self.hist_returns[-10:]).std() * np.sqrt(252)
        long = np.asarray(self.hist_returns[-40:]).std() * np.sqrt(252)
        if long <= 0:
            return 0.0
        return float(short / long - 1.0)

    # --- Layer A: Strategic (langzame baseline) ---
    def _strategic_multiplier(self, risk_score: float) -> float:
        if self._strategic_off:
            return 1.0
        return float(np.clip(1.0 - 0.9 * (risk_score / 100.0), 0.1, 1.0))

    # --- Layer B: Tactical (snellere aanpassingen) ---
    def _tactical_multiplier(self, vol_accel: float, drawdown: float) -> float:
        if self._tactical_off:
            return 1.0
        m = 1.0
        if vol_accel > 0.5:
            m *= float(np.clip(1.0 - (vol_accel - 0.5) / 1.5, 0.5, 1.0))
        if drawdown < -0.05:
            m *= float(np.clip(1.0 - (abs(drawdown) - 0.05) / 0.10, 0.5, 1.0))
        return m

    # --- Recovery multiplier ---
    def _recovery_multiplier(self, state: str) -> float:
        if self._recovery_off:
            return 1.0
        return {
            "NORMAL": 1.0, "ALERT": 0.7, "DEFENSIVE": 0.4,
            "STABILIZATION": 0.6, "RECOVERY": 0.8,
        }.get(state, 1.0)

    # --- Smoothing ---
    def _smooth(self, target: float) -> float:
        delta = target - self.prev_exposure
        if abs(delta) > self.max_daily_change:
            target = self.prev_exposure + np.sign(delta) * self.max_daily_change
        return float(np.clip(target, self.min_exposure, self.max_exposure))

    # --- Main entry point ---
    def approve(
        self,
        decision: RLRawDecision,
        portfolio: PortfolioState,
        equity: float,
        mc: MonteCarloResult | None = None,
        alpha_signals: dict | None = None,
    ) -> tuple[float, V21Breakdown]:
        """Bereken de effectieve exposure met volledige breakdown."""
        if decision.action.value == "sell" and decision.exit_reason.value != "none":
            self.prev_exposure = 0.0
            return 0.0, V21Breakdown(
                risk_score=0, opp_score=0, state=self.recovery.state,
                risk_budget=0, desired_exposure=0, effective_exposure=0,
                strategic_multiplier=1, tactical_multiplier=1,
                emergency_multiplier=1, recovery_multiplier=1,
                vol=0, drawdown=0, correlation=0, var_95=0, crash_prob=0,
                opp_breakdown={}, reason=f"exit: {decision.exit_reason.value}",
            )

        drawdown = self.update_drawdown(equity)
        vol = self._volatility()
        corr = self._correlation()
        vol_accel = self._vol_acceleration()
        var_95 = mc.var_95 if mc else 0.0
        crash_prob = mc.crash_probability if mc else 0.0
        daily_return = self.hist_returns[-1] if self.hist_returns else 0.0

        # --- Axis 1: Risk Environment Score ---
        risk_score = self.risk_score_calc.score(
            vol, self.vol_baseline, drawdown, corr, var_95, crash_prob, daily_return
        )

        # --- Axis 2: Opportunity Score (onafhankelijk) ---
        if self._opp_off:
            opp_score, opp_breakdown = 0.0, {"disabled": True}
        else:
            opp_score, opp_breakdown = self.opp_score_calc.score(
                self.hist_returns, vol, self.vol_baseline, drawdown, corr, alpha_signals
            )

        # --- Risk budget uit twee assen ---
        risk_budget = risk_budget_from_scores(risk_score, opp_score)
        state = state_label(risk_score, opp_score)

        # --- Recovery Engine ---
        recovery_state = self.recovery.update(drawdown, vol, self.vol_baseline,
                                              self.hist_returns)

        # --- Emergency Brake (Layer C) ---
        if not self._emergency_off:
            self.brake.check(daily_return, vol, self.vol_baseline)
        emergency_mult = self.brake.tick()

        # --- Combineer lagen ---
        strategic_mult = self._strategic_multiplier(risk_score)
        tactical_mult = self._tactical_multiplier(vol_accel, drawdown)
        recovery_mult = self._recovery_multiplier(recovery_state)

        desired = self.base_exposure
        effective = desired * risk_budget
        effective *= strategic_mult * tactical_mult * recovery_mult * emergency_mult
        effective = self._smooth(effective)

        reasons = []
        if risk_score > 50:
            reasons.append(f"risk={risk_score:.0f}")
        if opp_score < 30:
            reasons.append(f"opp={opp_score:.0f}")
        if emergency_mult < 1:
            reasons.append("EMERGENCY")
        if recovery_state != "NORMAL":
            reasons.append(f"recovery={recovery_state}")
        reason = ", ".join(reasons) if reasons else "normal"

        self.prev_exposure = effective
        return effective, V21Breakdown(
            risk_score=round(risk_score, 1), opp_score=round(opp_score, 1),
            state=state, risk_budget=round(risk_budget, 4),
            desired_exposure=round(desired, 4), effective_exposure=round(effective, 4),
            strategic_multiplier=round(strategic_mult, 4),
            tactical_multiplier=round(tactical_mult, 4),
            emergency_multiplier=round(emergency_mult, 4),
            recovery_multiplier=round(recovery_mult, 4),
            vol=round(vol, 4), drawdown=round(drawdown, 4),
            correlation=round(corr, 4), var_95=round(var_95, 4),
            crash_prob=round(crash_prob, 4), opp_breakdown=opp_breakdown,
            reason=reason,
        )

    def record_return(self, ret: float) -> None:
        self.hist_returns.append(ret)
        if len(self.hist_returns) > 500:
            self.hist_returns = self.hist_returns[-500:]
