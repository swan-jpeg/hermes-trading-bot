"""Train the RL decision layer with stable-baselines3 (PPO).

Run on your Windows PC (8GB VRAM / 16GB RAM is plenty — PPO is tiny):

    uv sync --extra ml --extra data
    uv run python -m hermes_bot.train_rl --ticker SPY --years 5 --timesteps 200000

The trained model is saved to `models/rl_ppo_<ticker>.zip`. Load it later with
`RLFusionModel(policy=TrainedPolicy(model))` — the policy interface is unchanged,
so the risk engine still gates every proposal.

Data: real prices via yfinance (free). If the network is unavailable it falls
back to synthetic prices so training still works offline.
"""
from __future__ import annotations

import argparse
import pathlib
import warnings

import numpy as np

from hermes_bot.portfolio import PortfolioState
from hermes_bot.rl import RulePolicy
from hermes_bot.rl.env import TradingEnv

warnings.filterwarnings("ignore")

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"


def load_prices(ticker: str, years: int) -> np.ndarray:
    """Load daily close prices via yfinance, or synthetic fallback."""
    try:
        import yfinance as yf

        df = yf.download(ticker, period=f"{years}y", progress=False, auto_adjust=True)
        closes = df["Close"].values.squeeze()
        closes = closes[~np.isnan(closes)]
        if len(closes) > 50:
            return closes.astype(float)
    except Exception:  # noqa: BLE001
        pass
    # Synthetic fallback (offline-safe): random walk with a mild drift.
    rng = np.random.default_rng(42)
    n = years * 252
    rets = rng.normal(0.0003, 0.012, n)
    return 100.0 * np.cumprod(1 + rets)


def build_fusion_signals(prices: np.ndarray, seed: int = 42) -> list[dict]:
    """Synthetic fusion signals (sentiment/certainty/quality) per day.

    In production these come from the fusion model; for training we generate
    plausible signals so the agent learns the mapping signal -> action.
    """
    rng = np.random.default_rng(seed)
    n = len(prices)
    # Sentiment loosely follows the price momentum.
    rets = np.diff(prices, prepend=prices[0]) / prices[0]
    mom = np.convolve(rets, np.ones(5) / 5, mode="same")
    signals = []
    for i in range(n):
        s = float(np.clip(mom[i] * 20, -1, 1)) + float(rng.normal(0, 0.1))
        signals.append({
            "sentiment": float(np.clip(s, -1, 1)),
            "zekerheid": float(np.clip(0.5 + abs(s) * 0.3, 0, 1)),
            "kwaliteit": float(np.clip(0.5 + mom[i] * 5, 0, 1)),
        })
    return signals


def make_env(prices: np.ndarray, signals: list[dict], seed: int = 42) -> TradingEnv:
    """Build the training environment with a fresh portfolio + rule policy."""
    portfolio = PortfolioState(cash=100_000.0)
    policy = RulePolicy()
    # The env doesn't need a real risk engine for training; risk gating happens
    # at inference. We pass a lightweight stub that returns neutral metrics.
    class _StubRisk:
        def __init__(self) -> None:
            self.var_95 = -0.02
            self.crash_prob = 0.05
            self.regime = "bull"

    return TradingEnv(
        prices=prices,
        fusion_signals=signals,
        portfolio_state=portfolio,
        risk_engine=_StubRisk(),
        rule_policy=policy,
        max_steps=len(prices),
        seed=seed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the RL decision layer (PPO)")
    ap.add_argument("--ticker", default="SPY", help="ticker voor training (default: SPY)")
    ap.add_argument("--years", type=int, default=5, help="jaren historie (default: 5)")
    ap.add_argument("--timesteps", type=int, default=200_000, help="PPO timesteps (default: 200k)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {args.ticker} ({args.years}y)...")
    prices = load_prices(args.ticker, args.years)
    signals = build_fusion_signals(prices, args.seed)
    print(f"  {len(prices)} datapunten geladen.")

    env = make_env(prices, signals, args.seed)
    print(f"Environment: obs={env.observation_space.shape}, act={env.action_space.n}")

    from stable_baselines3 import PPO

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        seed=args.seed,
    )

    print(f"Training PPO for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"rl_ppo_{args.ticker}.zip"
    model.save(str(path))
    print(f"\nModel opgeslagen: {path}")
    print("Gebruik het later met: RLFusionModel(policy=TrainedPolicy(model))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
