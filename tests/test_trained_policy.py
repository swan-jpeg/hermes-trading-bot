"""Tests for the trained-policy wrapper (SB3 integrated behind the interface).

Importing regular `stable_baselines3` HANGS on this server CPU (torch tries to
load cuda.bindings), so these tests stub PPO.load + predict and assert the
wrapper maps observations/actions correctly.
"""
from __future__ import annotations

import sys
import types

import numpy as np

from hermes_bot.rl.trained import TrainedPolicy
from hermes_bot.schemas import Action


class _FakeModel:
    """Minimal stand-in for a trained SB3 model."""

    def __init__(self, action: int) -> None:
        self._action = action

    def predict(self, obs, deterministic: bool = False):
        return np.array([self._action]), None


def _stub_sb3(predict_fn) -> None:
    """Inject a fake stable_baselines3 module with a PPO.load stub."""
    fake = types.ModuleType("fake_sb3")

    class _PPO:
        def __init__(self) -> None:
            raise NotImplementedError

        @staticmethod
        def load(path):
            return predict_fn()

    fake.PPO = _PPO
    sys.modules["stable_baselines3"] = fake


def _make_policy(action: int) -> TrainedPolicy:
    _stub_sb3(lambda: _FakeModel(action))
    return TrainedPolicy("models/rl_ppo_fake.zip")


def test_buy_action() -> None:
    """A model predicting BUY yields a buy decision with positive intent."""
    pol = _make_policy(1)
    d = pol.act({"entity": "SPY", "sentiment": 0.5, "zekerheid": 0.8,
                 "kwaliteit": 0.7, "regime": "bull", "current_price": 110.0})
    assert d.action == Action.BUY
    assert d.intent_to_alloc > 0.0


def test_sell_action() -> None:
    """A model predicting SELL yields a sell decision."""
    pol = _make_policy(2)
    d = pol.act({"entity": "SPY", "sentiment": -0.5, "zekerheid": 0.8,
                 "kwaliteit": 0.7, "regime": "bear", "current_price": 90.0})
    assert d.action == Action.SELL
    assert d.intent_to_alloc < 0.0


def test_hold_action() -> None:
    """A model predicting HOLD yields a hold decision with zero intent."""
    pol = _make_policy(0)
    d = pol.act({"entity": "SPY", "sentiment": 0.0, "zekerheid": 0.5,
                 "kwaliteit": 0.5, "regime": "unknown", "current_price": 100.0})
    assert d.action == Action.HOLD
    assert d.intent_to_alloc == 0.0


def test_observation_dimension() -> None:
    """The observation vector passed to the model is 9-dim."""
    captured: dict = {}

    class _Capture:
        def predict(self, obs, deterministic=False):
            captured["obs"] = obs
            return np.array([0]), None

    _stub_sb3(lambda: _Capture())
    pol = TrainedPolicy("models/rl_ppo_fake.zip")
    pol.act({"entity": "SPY", "sentiment": 0.2, "zekerheid": 0.6, "kwaliteit": 0.8,
             "regime": "crash", "current_price": 100.0})
    assert captured["obs"].shape == (9,)
