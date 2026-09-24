# Hermes Trading Bot

A modular multi-asset AI trading bot (Python). Goal: **good returns** and
**resilience against market crashes**, across stocks, ETFs, options, bonds and cash.

> ⚠️ **BACKTEST BEFORE MONEY, PAPER BEFORE LIVE.** This is a working skeleton with
> all architecture layers. The risk layer and backtests are mandatory gates
> before any live order.

## ⬇️ Download & run

[![Download](https://img.shields.io/badge/Download-Latest%20Release-0a84ff?style=for-the-badge&logo=github)](https://github.com/swan-jpeg/hermes-trading-bot/releases/latest)

Two download options per release:

| Option | What's inside | Size |
| --- | --- | --- |
| **Core** | Risk engine + backtest + portfolio + execution (the essential part) | small |
| **Full** | Everything, including the heavy-model extras (speech, emotion, pose) | large |

### Run it (any OS)

```bash
# 1. Download the zip (Core or Full) from the Releases page and unzip it.
# 2. Install Python 3.11+ and uv (https://docs.astral.sh/uv/).
# 3. Install dependencies:
uv sync --extra dev --extra ml --extra data
#    (Full only — heavy models, optional & large):
uv sync --extra multimodal
uv sync --extra video        # only on machines where mediapipe works

# 4. Run the demo:
uv run python -m hermes_bot.demo

# 5. Or start the web interface (port 9124):
uv run python -m hermes_bot.webui
```

On Windows, just double-click `install_windows.bat` (installs everything and runs the tests).

### Releases

| Version | Core (risk engine) | Full (everything) | Changes | Posted (Amsterdam) |
| --- | --- | --- | --- | --- |
| v0.3.0 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.3.0/hermes-trading-bot-v0.3.0-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.3.0/hermes-trading-bot-v0.3.0-full.zip) | Nieuwe impact agent (gratis LLM via Nous stepfun:free + skill voor eigen LLM) + geheel nieuwe architectuur: impact-agent → 44-dim AssetContext → RL-fusion → risk engine (regional/regime als risk-inputs) + replay-module voor historische training | 2026-09-24 08:27 |
| v0.2.5 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.5/hermes-trading-bot-v0.2.5-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.5/hermes-trading-bot-v0.2.5-full.zip) | Default device = CPU (stabiel); DirectML alleen met --device dml | 2026-09-23 18:10 |
| v0.2.4 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.4/hermes-trading-bot-v0.2.4-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.4/hermes-trading-bot-v0.2.4-full.zip) | Fix resolve_device: alle branches geven torch.device-object terug | 2026-09-23 17:50 |
| v0.2.3 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.3/hermes-trading-bot-v0.2.3-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.3/hermes-trading-bot-v0.2.3-full.zip) | Fix --device dml (DirectML device-object voor SB3) | 2026-09-23 17:28 |
| v0.2.2 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.2/hermes-trading-bot-v0.2.2-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.2/hermes-trading-bot-v0.2.2-full.zip) | `--device` optie (AMD Radeon via DirectML) | 2026-09-23 12:34 |
| v0.2.1 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.1/hermes-trading-bot-v0.2.1-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.1/hermes-trading-bot-v0.2.1-full.zip) | Fix RL reward-bug (turnover-penalty per stap) | 2026-09-23 12:20 |
| v0.2.0 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.0/hermes-trading-bot-v0.2.0-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.2.0/hermes-trading-bot-v0.2.0-full.zip) | Werkende TradingEnv + train_rl.py (PPO) | 2026-09-23 11:45 |
| v0.1.0 | [core](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.1.0/hermes-trading-bot-v0.1.0-core.zip) | [full](https://github.com/swan-jpeg/hermes-trading-bot/releases/download/v0.1.0/hermes-trading-bot-v0.1.0-full.zip) | Eerste release (core + full) | 2026-09-23 10:15 |
<!-- RELEASES -->

## Architecture (7 layers)

```
24/7 data → signals (audio/video/reports/alerts) → AI agents → fusion
         → RL decision + Monte Carlo/risk → execution (paper/live)
```

See `PLAN-QWEN.md` for the implementation plan.

## Quick start

```bash
# 1. Install (Python 3.11+)
uv sync --extra dev --extra ml --extra data

# 2. Heavy multimodal models (optional, large — see below)
uv sync --extra multimodal
uv sync --extra video        # only on machines where mediapipe works

# 3. Tests + quality
uv run pytest
uv run ruff check hermes_bot tests

# 4. End-to-end demo
uv run python -m hermes_bot.demo

# 5. Web interface (port 9124)
uv run python -m hermes_bot.webui
```

## Train the RL model (on your PC)

The RL decision layer is a Gymnasium environment + PPO (stable-baselines3).
Train it on your own machine — a small PPO model needs only a few MB, so
**8GB VRAM / 16GB RAM is plenty** (even CPU works).

```bash
# 1. Install with the ML extras (includes stable-baselines3 + torch):
uv sync --extra ml --extra data

# 2. Train (real SPY data via yfinance, or synthetic offline):
uv run python -m hermes_bot.train_rl --ticker SPY --years 5 --timesteps 200000

# 3. The model is saved to models/rl_ppo_SPY.zip
```

**AMD Radeon GPU on Windows?** PyTorch uses CUDA (NVIDIA) by default, so a
Radeon is not used automatically. Install PyTorch DirectML and pass `--device dml`:

```bash
uv pip install torch-directml
uv run python -m hermes_bot.train_rl --device dml --ticker SPY --years 5 --timesteps 200000
```

> Note: DirectML is a public preview and for this tiny PPO model it is often
> NOT faster than CPU. If `--device dml` is slower, just drop it and train on CPU.

The trained model plugs into the existing decision layer behind the same
interface — the risk engine still gates every proposal (RL proposes, Risk approves).

## Heavy models (optional)

The multimodal analysis (speech→text, emotion, facial expression, pose) uses
**existing open-source models** — you don't need to train anything yourself:

| Component | Model | Extra |
|---|---|---|
| Speech→text | faster-whisper | `multimodal` |
| Emotion (speech) | prosody-based (librosa) | `multimodal` |
| Facial expression | DeepFace | `multimodal` |
| Pose/body language | MediaPipe Pose | `video` |
| Financial sentiment | FinBERT (transformers) | `multimodal` |

**Important:** the heavy models are only loaded when `use_heavy_models=True`
in `AudioAnalyzer`/`VideoAnalyzer`. Off by default (offline-safe) so the pipeline
stays fast and stable. Enable on a machine where the models work (e.g. Windows
with GPU). MediaPipe can hard-crash on some CPUs — install the `video` extra
only where it works.

## Directory structure

```
hermes_bot/
├── data/            # LAYER 1-2: collectors, storage, features
├── signals/         # LAYER 3: audio/video, reports, alerts, sentiment
├── agents/          # LAYER 4: fundamental AI agent
├── fusion/          # LAYER 5: fusion model (quality/certainty/emotion)
├── rl/              # LAYER 6: RL decision (env, reward, policy)
├── simulation/      # LAYER 6: Monte Carlo (bootstrap, scenarios, metrics)
├── risk/            # LAYER 6: risk engine (sizing, drawdown, stress, hedging)
├── portfolio/       # allocation across asset classes (multi-asset)
├── execution/       # LAYER 7: output action → broker (paper/live)
├── expansions/      # B2B bottleneck, regime detection, orderflow
└── webui.py         # web interface (port 9124)
```

## Core design rules

1. **RL proposes, Risk approves** — never send an RL output straight to the market.
2. **Fail-closed**: if the broker is unreachable → no order.
3. **Decision ≠ execution**: log intent and real fills.
4. **Verify every order** with the broker.
5. **Backtest before money, paper before live.**

## Windows / GitHub

- This project is ready to push to GitHub (see `git remote add origin ...`).
- On Windows: install Python 3.11+, `uv`, and run the same commands.
  The `video` extra (mediapipe) usually works well on Windows.
- API keys go in `.env` (never in code). See `.env.example`.

## License

MIT — see `LICENSE`.
