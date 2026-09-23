"""Tests for the RL environment (T6) — rich AssetContext observation."""
from __future__ import annotations

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.rl import RulePolicy
from hermes_bot.rl.context import AssetContext
from hermes_bot.rl.env import TradingEnv

# Indexen in de 44-dim observatie (zie AssetContext._OBS_KEYS).
# Portfolio/risico-velden staan achteraan.
_IDX_ALLOC = 41
_IDX_PNL = 39
_IDX_SENTIMENT = 0  # impact_direction (categorie 1, kern)


def _make_env(n: int = 100, contexts: list[AssetContext] | None = None) -> TradingEnv:
    prices = np.linspace(100, 150, n)
    signals = [{"sentiment": 0.5, "zekerheid": 0.8, "kwaliteit": 0.7}] * n
    if contexts is None:
        contexts = [AssetContext(impact_direction=0.5)] * n
    return TradingEnv(
        prices=prices,
        fusion_signals=signals,
        portfolio_state=PortfolioState(cash=100_000.0),
        risk_engine=None,
        rule_policy=RulePolicy(),
        max_steps=n,
        asset_contexts=contexts,
    )


def test_env_observation_shape() -> None:
    """The observation is the rich AssetContext vector (44-dim)."""
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == (44,)
    assert obs.dtype == np.float32


def test_env_observation_values() -> None:
    """The impact-direction (categorie 1) comes from the AssetContext."""
    ctx = AssetContext(impact_direction=0.8)
    env = _make_env(contexts=[ctx] * 100)
    obs, _ = env.reset()
    assert abs(float(obs[_IDX_SENTIMENT]) - 0.8) < 1e-6  # impact_direction


def test_env_buy_sets_allocation() -> None:
    """A BUY action opens a position (allocation > 0)."""
    env = _make_env()
    env.reset()
    obs, _reward, _done, _trunc, _info = env.step(1)  # BUY
    assert float(obs[_IDX_ALLOC]) > 0.0  # allocation


def test_env_sell_closes_position() -> None:
    """A SELL action closes the position (allocation -> 0)."""
    env = _make_env()
    env.reset()
    env.step(1)  # BUY
    obs, _reward, _done, _trunc, _info = env.step(2)  # SELL
    assert float(obs[_IDX_ALLOC]) == 0.0  # allocation


def test_env_pnl_tracks_price() -> None:
    """Unrealized P&L rises as the price rises while holding."""
    env = _make_env()
    env.reset()
    env.step(1)  # BUY at ~100
    for _ in range(20):
        obs, _r, _d, _t, _i = env.step(0)  # HOLD
    assert float(obs[_IDX_PNL]) > 0.0  # positive P&L


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
    contexts = [AssetContext(impact_direction=0.5)] * n
    env = TradingEnv(
        prices=prices,
        fusion_signals=signals,
        portfolio_state=PortfolioState(cash=100_000.0),
        risk_engine=None,
        rule_policy=RulePolicy(),
        max_steps=n,
        asset_contexts=contexts,
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


def test_asset_context_observation_dim() -> None:
    """AssetContext.observation() matches the env's observation space."""
    ctx = AssetContext()
    env = _make_env()
    assert ctx.obs_dim == env.observation_space.shape[0]
    assert ctx.observation().shape == (44,)
