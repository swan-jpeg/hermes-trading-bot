"""LAAG 6: RL-besluitvorming. [[07]] — RL voorstelt, Risk keurt goed.

Implementatie: eenvoudige regel-gebaseerde policy als warmstart + de
reward-functie (rendement - drawdown-straf - turnover). De policy is
vervangbaar door een getrainde PPO/SAC via stable-baselines3.

Architectuurverbetering: de policy is nu PRIJS- en POSITIE-bewust. Hij
ziet de huidige positie (entry, ongerealiseerde winst/verlies) en de
huidige prijs, zodat hij expliciet winst kan nemen (take-profit),
verlies kan beperken (stop-loss) en trailing stops kan toepassen —
i.p.v. alleen te reageren op sentiment.
"""
from __future__ import annotations

from datetime import UTC, datetime

from hermes_bot.portfolio import PortfolioState
from hermes_bot.schemas import Action, ExitReason, Position, RLRawDecision


class BaseRLPolicy:
    """Interface for the RL decision layer."""

    def act(self, state: dict) -> RLRawDecision:
        raise NotImplementedError


class RulePolicy(BaseRLPolicy):
    """Warmstart-policy: prijs-bewust + sentiment.

    Prioriteit (belangrijkste eerst):
    1. EXIT-regels op bestaande posities (take-profit / stop-loss / trailing).
    2. Sentiment-gestuurde BUY/SELL op nieuwe of bestaande posities.
    """

    def act(self, state: dict) -> RLRawDecision:
        entity = state["entity"]
        sentiment = state.get("sentiment", 0.0)
        zekerheid = state.get("zekerheid", 0.0)
        threshold = state.get("threshold", 0.3)
        current_price = state.get("current_price", 0.0)
        position: Position | None = state.get("position")
        peak = state.get("peak", current_price)

        # --- 1. Exit rules on existing positions (take profit / limit loss).
        if position is not None and current_price > 0:
            reason = position.exit_reason_at(current_price, peak)
            if reason != ExitReason.NONE:
                # intent_to_alloc is a portfolio weight (-1..+1), not an
                # absolute qty. Close the position fully (weight -1 = 100%).
                pnl = position.unrealized_pnl_pct(current_price)
                return RLRawDecision(
                    entity=entity,
                    action=Action.SELL,
                    intent_to_alloc=-1.0,
                    zekerheid=round(zekerheid, 4),
                    rationale=f"exit: {reason.value} (pnl={pnl:.2%})",
                    timestamp=datetime.now(UTC),
                    exit_reason=reason,
                )

        # --- 2. Sentiment-driven direction on new/existing positions.
        if sentiment > threshold and zekerheid > 0.5:
            action, alloc = Action.BUY, min(0.1, sentiment * zekerheid)
        elif sentiment < -threshold and zekerheid > 0.5:
            # Sell (or reduce) on negative sentiment.
            size = min(0.1, abs(sentiment) * zekerheid)
            if position is not None:
                size = min(size, position.qty)
            action, alloc = Action.SELL, -size
        else:
            action, alloc = Action.HOLD, 0.0

        return RLRawDecision(
            entity=entity,
            action=action,
            intent_to_alloc=round(alloc, 4),
            zekerheid=round(zekerheid, 4),
            rationale=f"rule-policy: sentiment={sentiment:.2f}, zekerheid={zekerheid:.2f}",
            timestamp=datetime.now(UTC),
            exit_reason=ExitReason.NONE,
        )


class RLFusionModel:
    """Centrale beslissingslaag: state = fusie + positie + regime + prijs."""

    def __init__(self, policy: BaseRLPolicy | None = None) -> None:
        self.policy = policy or RulePolicy()

    def decide(
        self,
        fusion: dict,
        entity_state: dict,
        portfolio_state: PortfolioState,
        current_price: float = 0.0,
    ) -> RLRawDecision:
        """Build the state vector and let the policy make a proposal.

        Dit voorstel gaat ALTIJD eerst door de risico-engine.
        portfolio_state is nu een PortfolioState (niet dict) zodat de policy
        de open positie + entry-prijs ziet en winst kan nemen.
        """
        entity = fusion.get("entity_id", entity_state.get("entity_id", "?"))
        position = portfolio_state.positions.get(entity)
        peak = portfolio_state.peak_for(entity, current_price) if position else current_price

        state = {
            "entity": entity,
            "sentiment": fusion.get("emotie", {}).get("sentiment", 0.0),
            "zekerheid": fusion.get("zekerheid", 0.0),
            "kwaliteit": fusion.get("kwaliteit", 0.0),
            "regime": portfolio_state.regime,
            "current_price": current_price,
            "position": position,
            "peak": peak,
        }
        return self.policy.act(state)


def compute_reward(
    returns: list[float],
    drawdowns: list[float],
    turnover: float,
    lam_dd: float = 0.5,
    lam_to: float = 0.02,
) -> float:
    """Reward = rendement - drawdown-straf - turnover-straf.

    Balanceert winst (returns) tegen crashweerstand (drawdown) en
    overtraden (turnover).
    """
    r = float(sum(returns))
    dd_penalty = lam_dd * max(0.0, -min(drawdowns)) if drawdowns else 0.0
    to_penalty = lam_to * turnover
    return r - dd_penalty - to_penalty