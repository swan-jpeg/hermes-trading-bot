"""LAYER 6: RL environment per spec (section 4).

A working Gymnasium environment for training the RL decision layer with
stable-baselines3. The observation vector is built from REAL values (fusion
signal, portfolio state, Monte Carlo risk) and the reward uses the existing
`compute_reward` function (return - drawdown penalty - turnover penalty).

The environment steps through a price series; each step the agent sees the
current market/fusion/portfolio state, proposes an action, and the environment
applies it to the portfolio and returns the reward.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hermes_bot.rl import compute_reward
from hermes_bot.rl.context import AssetContext
from hermes_bot.schemas import Action

if TYPE_CHECKING:
    from hermes_bot.portfolio import PortfolioState

# Action mapping: 0=HOLD, 1=BUY, 2=SELL
_ACTION_TO_ENUM = {0: Action.HOLD, 1: Action.BUY, 2: Action.SELL}


class TradingEnv(gym.Env):
    """Gymnasium environment for trading RL.

    Observation (9-dim, per section 4.1):
        0: sentiment (fusion) -1..1
        1: certainty (fusion) 0..1
        2: quality (fusion) 0..1
        3: regime 0..3 (bull/bear/highvol/crash)
        4: unrealized P&L % -1..+5
        5: holding days 0..N
        6: current allocation 0..1
        7: VaR95 (portfolio) -1..0
        8: crash probability 0..1

    Action: discrete (HOLD/BUY/SELL). Reward: compute_reward.
    """

    def __init__(
        self,
        prices: np.ndarray,
        fusion_signals: list[dict],
        portfolio_state: PortfolioState,
        risk_engine,
        rule_policy,
        max_steps: int = 1000,
        seed: int = 42,
        asset_contexts: list[AssetContext] | None = None,
    ) -> None:
        super().__init__()
        self.prices = np.asarray(prices, dtype=float)
        self.fusion_signals = fusion_signals
        self.portfolio = portfolio_state
        self.risk_engine = risk_engine
        self.rule_policy = rule_policy
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        # Rich per-asset context (impact/fusion/bottleneck/source). If not
        # given, build a neutral AssetContext per step from fusion_signals.
        self.asset_contexts = asset_contexts or [
            AssetContext() for _ in range(max(len(prices), 1))
        ]

        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL
        self.obs_dim = self.asset_contexts[0].obs_dim if self.asset_contexts else 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.entry_price: float | None = None
        self.entry_step: int | None = None
        self.peak_price: float = 0.0
        self.holding_days = 0
        self.allocation = 0.0
        self.var_95 = 0.0
        self.crash_prob = 0.0
        self.regime_code = 0.0
        self._returns: list[float] = []
        self._drawdowns: list[float] = []
        self._turnover = 0.0
        self._step_turnover = 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.entry_price = None
        self.entry_step = None
        self.peak_price = 0.0
        self.holding_days = 0
        self.allocation = 0.0
        self.var_95 = 0.0
        self.crash_prob = 0.0
        self.regime_code = 0.0
        self._returns = []
        self._drawdowns = []
        self._turnover = 0.0
        self._step_turnover = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        """Take a step in the environment."""
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {action}")

        price = self._current_price()
        prev_price = self.prices[max(0, self.current_step - 1)] if self.current_step > 0 else price

        # Apply the action to the portfolio.
        self._apply_action(int(action), price)

        # Daily return of the portfolio (allocation * price move).
        price_ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        port_ret = self.allocation * price_ret
        self._returns.append(port_ret)

        # Track drawdown from peak equity.
        equity = 1.0 + float(np.sum(self._returns))
        self.peak_price = max(self.peak_price, equity)
        dd = equity / self.peak_price - 1.0 if self.peak_price > 0 else 0.0
        self._drawdowns.append(dd)

        # Advance the step counter.
        self.current_step += 1
        if self.entry_price is not None:
            self.holding_days += 1

        # Reward via the existing compute_reward function.
        # Use ONLY this step's turnover (not the cumulative) so the penalty
        # is charged once when a trade happens, not on every following step.
        reward = compute_reward(
            returns=[port_ret],
            drawdowns=[self._drawdowns[-1]] if self._drawdowns else [0.0],
            turnover=self._step_turnover,
        )
        self._step_turnover = 0.0

        done = self.current_step >= self.max_steps or self.current_step >= len(self.prices) - 1
        return self._get_observation(), float(reward), done, False, {}

    def _apply_action(self, action: int, price: float) -> None:
        """Apply the action to the portfolio state."""
        if action == 1:  # BUY
            if self.entry_price is None:
                self.entry_price = price
                self.entry_step = self.current_step
                self.holding_days = 0
                self.allocation = 0.5  # half position to start
                self._step_turnover += 0.5
                self._turnover += 0.5
        elif action == 2:  # SELL
            if self.entry_price is not None:
                self.allocation = 0.0
                self.entry_price = None
                self.entry_step = None
                self.holding_days = 0
                self._step_turnover += 0.5
                self._turnover += 0.5

    def _current_price(self) -> float:
        return float(self.prices[min(self.current_step, len(self.prices) - 1)])

    def _get_observation(self) -> np.ndarray:
        """Build the rich observation vector from the AssetContext.

        De AssetContext draagt de impact-vector (categorie 1), bron-info
        (categorie 2), relevance (categorie 3), marktinterpretatie (categorie
        4), bottleneck (categorie 5) en tijd (categorie 6). De portfolio/risico
        velden (regime, P&L, holding, allocatie, VaR, crash) worden hier
        live ingevuld.
        """
        ctx = self._current_context()

        # Unrealized P&L % if holding.
        pnl = 0.0
        if self.entry_price and self.entry_price > 0:
            pnl = (self._current_price() - self.entry_price) / self.entry_price

        ctx.update(
            regime_code=self.regime_code,
            pnl=pnl,
            holding_days=float(self.holding_days),
            allocation=self.allocation,
            var_95=self.var_95,
            crash_prob=self.crash_prob,
        )
        return ctx.observation()

    def _current_fusion(self) -> dict:
        """Get the fusion signal for the current step (or a neutral default)."""
        if self.fusion_signals and self.current_step < len(self.fusion_signals):
            return self.fusion_signals[self.current_step]
        return {"sentiment": 0.0, "zekerheid": 0.5, "kwaliteit": 0.5}

    def _current_context(self) -> AssetContext:
        """Get the AssetContext for the current step (or a neutral default)."""
        if self.asset_contexts and self.current_step < len(self.asset_contexts):
            return self.asset_contexts[self.current_step]
        return AssetContext()

    def set_risk_metrics(self, var_95: float, crash_prob: float, regime: str) -> None:
        """Inject Monte Carlo / regime values from the risk engine (per step)."""
        self.var_95 = float(var_95)
        self.crash_prob = float(crash_prob)
        regime_map = {"bull": 0.0, "bear": 1.0, "highvol": 2.0, "crash": 3.0}
        self.regime_code = regime_map.get(regime, 0.0)
