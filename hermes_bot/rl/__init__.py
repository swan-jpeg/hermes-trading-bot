"""LAAG 6: RL-besluitvorming. [[07]] — RL voorstelt, Risk keurt goed.

Implementatie: eenvoudige regel-gebaseerde policy als warmstart + de
reward-functie (rendement - drawdown-straf - turnover). De policy is
vervangbaar door een getrainde PPO/SAC via stable-baselines3.
"""
from __future__ import annotations

from datetime import UTC, datetime

from hermes_bot.schemas import Action, RLRawDecision


class BaseRLPolicy:
    """Interface voor de RL-beslissingslaag."""

    def act(self, state: dict) -> RLRawDecision:
        raise NotImplementedError


class RulePolicy(BaseRLPolicy):
    """Warmstart-policy: koop bij positief sentiment+zekerheid, verkoop bij negatief.

    Dit is een deterministische baseline; vervang door getrainde RL.
    """

    def act(self, state: dict) -> RLRawDecision:
        entity = state["entity"]
        sentiment = state.get("sentiment", 0.0)
        zekerheid = state.get("zekerheid", 0.0)
        threshold = state.get("threshold", 0.3)

        if sentiment > threshold and zekerheid > 0.5:
            action, alloc = Action.BUY, min(0.1, sentiment * zekerheid)
        elif sentiment < -threshold and zekerheid > 0.5:
            action, alloc = Action.SELL, -min(0.1, abs(sentiment) * zekerheid)
        else:
            action, alloc = Action.HOLD, 0.0

        return RLRawDecision(
            entity=entity,
            action=action,
            intent_to_alloc=round(alloc, 4),
            zekerheid=round(zekerheid, 4),
            rationale=f"rule-policy: sentiment={sentiment:.2f}, zekerheid={zekerheid:.2f}",
            timestamp=datetime.now(UTC),
        )


class RLFusionModel:
    """Centrale beslissingslaag: state = fusie + agent + regionaal + regime."""

    def __init__(self, policy: BaseRLPolicy | None = None) -> None:
        self.policy = policy or RulePolicy()

    def decide(
        self,
        fusion: dict,
        entity_state: dict,
        portfolio_state: dict,
    ) -> RLRawDecision:
        """Bouw state-vector en laat policy een voorstel doen.
        Dit voorstel gaat ALTIJD eerst door de risico-engine.
        """
        state = {
            "entity": fusion.get("entity_id", entity_state.get("entity_id", "?")) ,
            "sentiment": fusion.get("emotie", {}).get("sentiment", 0.0),
            "zekerheid": fusion.get("zekerheid", 0.0),
            "kwaliteit": fusion.get("kwaliteit", 0.0),
            "regime": portfolio_state.get("regime", "unknown"),
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
