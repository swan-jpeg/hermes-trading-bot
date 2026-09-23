"""Integrate a trained PPO model into the decision layer.

`TrainedPolicy` wraps a stable-baselines3 model behind the SAME `BaseRLPolicy`
interface as `RulePolicy`, so the bot can swap a trained model in without
changing the decision layer or bypassing the risk engine.

The trained model (e.g. `models/rl_ppo_SPY.zip`) is produced by `train_rl.py`
on the user's PC. Load it here with:

    TrainedPolicy("models/rl_ppo_SPY.zip")

Import of stable-baselines3 is LAZY (only when a model path is given) so the
rest of the bot keeps working on machines where SB3 is not installed or
where `import stable_baselines3` hangs (server CPUs).
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hermes_bot.rl import BaseRLPolicy
from hermes_bot.schemas import Action, ExitReason, RLRawDecision


class TrainedPolicy(BaseRLPolicy):
    """A trained stable-baselines3 PPO model as a BaseRLPolicy.

    Builds the same 9-dim observation as the training environment and maps
    the model's discrete action (0=HOLD, 1=BUY, 2=SELL) to an RLRawDecision.
    SB3 is imported lazily so this class is importable without SB3 installed.
    """

    _ACTION = {0: Action.HOLD, 1: Action.BUY, 2: Action.SELL}

    def __init__(self, model_path: str | Path) -> None:
        from stable_baselines3 import PPO  # lazy import

        self.model = PPO.load(str(model_path))
        # Align the observation dimension with the training environment.
        self._obs_dim = 9

    def act(self, state: dict) -> RLRawDecision:
        entity = state.get("entity", "?")
        sentiment = float(state.get("sentiment", 0.0))
        zekerheid = float(state.get("zekerheid", 0.0))
        kwaliteit = float(state.get("kwaliteit", 0.0))
        regime = str(state.get("regime", "unknown"))
        current_price = float(state.get("current_price", 0.0))
        position = state.get("position")
        peak = float(state.get("peak", current_price))

        pnl = 0.0
        alloc = 0.0
        if position is not None and current_price > 0:
            pnl = position.unrealized_pnl_pct(current_price)
            alloc = min(1.0, position.qty)

        obs = np.array([
            sentiment,                                            # 0
            zekerheid,                                            # 1
            kwaliteit,                                            # 2
            self._regime_code(regime),                            # 3
            pnl,                                                  # 4
            0.0,                     # 5 holding days (not tracked)
            alloc,                                                # 6
            0.0,                     # 7 VaR95 (filled by risk layer)
            0.0,                     # 8 crash prob (filled by risk layer)
        ], dtype=np.float32)

        action_n, _states = self.model.predict(obs, deterministic=False)
        action = self._ACTION[int(action_n)]

        exit_reason = ExitReason.NONE
        if position is not None and current_price > 0:
            exit_reason = position.exit_reason_at(current_price, peak)
            if action == Action.SELL or exit_reason != ExitReason.NONE:
                # Closing on a price-driven exit.
                if exit_reason != ExitReason.NONE:
                    action = Action.SELL

        if action == Action.BUY:
            intent = 0.5
        elif action == Action.SELL:
            intent = -1.0 if (position is not None) else -0.5
        else:
            intent = 0.0

        return RLRawDecision(
            entity=entity,
            action=action,
            intent_to_alloc=round(float(intent), 4),
            zekerheid=round(zekerheid, 4),
            rationale=f"trained-policy: action={action.value}",
            timestamp=datetime.now(UTC),
            exit_reason=exit_reason,
        )

    @staticmethod
    def _regime_code(regime: str) -> float:
        return {"bull": 0.0, "bear": 1.0, "highvol": 2.0, "crash": 3.0}.get(regime, 0.0)
