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

| Version | Core (risk engine) | Full (everything) |
| --- | --- | --- |
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
