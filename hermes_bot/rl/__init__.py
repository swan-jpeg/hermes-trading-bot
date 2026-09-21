"""LAAG 6: RL-besluitvorming. [[07]] — RL voorstelt, Risk keurt goed."""
from __future__ import annotations

from hermes_bot.schemas import RLRawDecision


class BaseRLPolicy:
    """Interface voor de RL-beslissingslaag."""

    def act(self, state: dict) -> RLRawDecision:
        raise NotImplementedError


class RLFusionModel:
    """Centrale beslissingslaag: state = fusie + agent + regionaal + regime."""

    def __init__(self, policy: BaseRLPolicy | None = None) -> None:
        self.policy = policy

    def decide(
        self,
        fusion: dict,
        entity_state: dict,
        portfolio_state: dict,
    ) -> RLRawDecision:
        """Bouw state-vector en laat policy een voorstel doen.
        Dit voorstel gaat ALTIJD eerst door de risico-engine.
        """
        raise NotImplementedError
