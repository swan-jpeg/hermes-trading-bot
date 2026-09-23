"""Tests for the RL environment (T6)."""
from __future__ import annotations

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.rl import RulePolicy
from hermes_bot.rl.env import TradingEnv


def _make_env(n: int = 100) -> TradingEnv:
    prices = np.linspace(100, 150, n)
    signals = [{"sentiment": 0.5, "zekerheid": 0.8, "kwaliteit": 0.7}] * n
    return TradingEnv(
        prices=prices,
        fusion_signals=signals,
        portfolio_state=PortfolioState(cash=100_000.0),
        risk_engine=None,
        rule_policy=RulePolicy(),
        max_steps=n,
    )


def test_env_observation_shape() -> None:
    """The observation is a 9-dim vector."""
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == (9,)
    assert obs.dtype == np.float32


def test_env_observation_values() -> None:
    """Sentiment/certainty/quality come from the fusion signal."""
    env = _make_env()
    obs, _ = env.reset()
    assert abs(float(obs[0]) - 0.5) < 1e-6  # sentiment
    assert abs(float(obs[1]) - 0.8) < 1e-6  # certainty
    assert abs(float(obs[2]) - 0.7) < 1e-6  # quality


def test_env_buy_sets_allocation() -> None:
    """A BUY action opens a position (allocation > 0)."""
    env = _make_env()
    env.reset()
    obs, _reward, _done, _trunc, _info = env.step(1)  # BUY
    assert float(obs[6]) > 0.0  # allocation


def test_env_sell_closes_position() -> None:
    """A SELL action closes the position (allocation -> 0)."""
    env = _make_env()
    env.reset()
    env.step(1)  # BUY
    obs, _reward, _done, _trunc, _info = env.step(2)  # SELL
    assert float(obs[6]) == 0.0  # allocation


def test_env_pnl_tracks_price() -> None:
    """Unrealized P&L rises as the price rises while holding."""
    env = _make_env()
    env.reset()
    env.step(1)  # BUY at ~100
    for _ in range(20):
        obs, _r, _d, _t, _i = env.step(0)  # HOLD
    assert float(obs[4]) > 0.0  # positive P&L


def test_env_episode_ends() -> None:
    """The episode ends at max_steps."""
    env = _make_env(n=10)
    env.reset()
    done = False
    for _ in range(20):
        _obs, _r, done, _t, _i = env.step(0)
        if done:
            break
    assert done


def test_env_invalid_action_raises() -> None:
    """An invalid action raises ValueError."""
    env = _make_env()
    env.reset()
    try:
        env.step(99)
        raise AssertionError("moest ValueError geven")
    except ValueError:
        pass


def test_env_reward_is_float() -> None:
    """The reward is a finite float."""
    env = _make_env()
    env.reset()
    env.step(1)
    _obs, reward, _d, _t, _i = env.step(0)
    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_non_negative_hold_reward_on_rise() -> None:
    """Regression: a HOLD during a rising price must not earn a negative reward.

    The old env charged the cumulative turnover penalty every step, so on a
    smooth upward line it returned ~-0.01/day and the episode summed to -684.
    Fix: the turnover penalty is charged only on the step that trades.
    """
    n = 60
    prices = np.linspace(100, 130, n)  # monotonic rise, no drawdown
    signals = [{"sentiment": 0.5, "zekerheid": 0.8, "kwaliteit": 0.7}] * n
    env = TradingEnv(
        prices=prices,
        fusion_signals=signals,
        portfolio_state=PortfolioState(cash=100_000.0),
        risk_engine=None,
        rule_policy=RulePolicy(),
        max_steps=n,
    )
    env.reset()
    env.step(1)  # BUY on day 0
    # The buy day itself has a small turnover cost; subsequent holds should
    # earn a positive reward as price rises.
    hold_rewards = []
    for _ in range(n - 1):
        _obs, r, done, _t, _i = env.step(0)
        if not done:
            hold_rewards.append(r)
    assert len(hold_rewards) > 40
    assert sum(hold_rewards) > 0.0  # rising market, holds accumulate profit
    assert all(r >= -1e-9 for r in hold_rewards[1:])  # no per-step turnover drain

