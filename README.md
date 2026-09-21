# Hermes Trading Bot

Modulaire multi-asset AI-trading-bot (Python). Doel: **goede winst** én
**weerstand tegen marktcrashes**, over aandelen, ETF's, opties, obligaties en cash.

> ⚠️ **BACKTEST VÓÓR GELD, PAPER VÓÓR LIVE.** Dit is een werkend skeleton met
> alle architectuurlagen. De risicolaag en backtests zijn verplichte poorten
> vóór enige live order.

## Architectuur (7 lagen)

```
24/7 data → signalen (audio/video/reports/alerts) → AI-agenten → fusie
         → RL-besluitvorming + Monte Carlo/risico → executie (paper/live)
```

Zie `~/obsidian-vault/TradingBot/` voor de volledige architectuurnotes, of
`PLAN-QWEN.md` voor het implementatieplan.

## Snelle start

```bash
# 1. Installeer (Python 3.11+)
uv sync --extra dev --extra ml --extra data

# 2. Zware multimodale modellen (optioneel, groot — zie hieronder)
uv sync --extra multimodal
uv sync --extra video        # alleen op machines waar mediapipe werkt

# 3. Tests + kwaliteit
uv run pytest
uv run ruff check hermes_bot tests

# 4. End-to-end demo
uv run python -m hermes_bot.demo

# 5. Web-interface (poort 9124)
uv run python -m hermes_bot.webui
```

## Zware modellen (optioneel)

De multimodale analyse (spraak→tekst, emotie, gezichtsuitdrukking, pose) gebruikt
**bestaande opensource-modellen** — je hoeft niets zelf te trainen:

| Component | Model | Extra |
|---|---|---|
| Spraak→tekst | faster-whisper | `multimodal` |
| Emotie (spraak) | prosodie-gebaseerd (librosa) | `multimodal` |
| Gezichtsuitdrukking | DeepFace | `multimodal` |
| Pose/lichaamstaal | MediaPipe Pose | `video` |
| Financieel sentiment | FinBERT (transformers) | `multimodal` |

**Belangrijk:** de zware modellen worden alleen geladen als `use_heavy_models=True`
in `AudioAnalyzer`/`VideoAnalyzer`. Standaard uit (offline-safe) zodat de pipeline
snel en stabiel blijft. Zet aan op een machine waar de modellen werken (bv. Windows
met GPU). MediaPipe geeft op sommige CPU's een harde crash — installeer de `video`-extra
alleen waar het werkt.

## Mappenstructuur

```
hermes_bot/
├── data/            # LAAG 1-2: collectors, storage, features
├── signals/         # LAAG 3: audio/video, reports, alerts, sentiment
├── agents/          # LAAG 4: fundamentele AI-agent
├── fusion/          # LAAG 5: fusiemodel (kwaliteit/zekerheid/emotie)
├── rl/              # LAAG 6: RL-besluitvorming (env, reward, policy)
├── simulation/      # LAAG 6: Monte Carlo (bootstrap, scenario's, metrics)
├── risk/            # LAAG 6: risico-engine (sizing, drawdown, stress, hedging)
├── portfolio/       # allocatie over assetklassen (multi-asset)
├── execution/       # LAAG 7: output actie → broker (paper/live)
├── expansions/      # B2B bottleneck, regime-detectie, orderflow
└── webui.py         # web-interface (poort 9124)
```

## Kernontwerpregels

1. **RL voorstelt, Risk keurt goed** — nooit een RL-output rechtstreeks naar de markt.
2. **Fail-closed**: als broker onbereikbaar is → géén order.
3. **Decision ≠ executie**: log intentie én echte fills.
4. **Verifieer elke order** met de broker.
5. **Backtest vóór geld, paper vóór live.**

## Windows / GitHub

- Dit project is klaar om naar GitHub te pushen (zie `git remote add origin ...`).
- Op Windows: installeer Python 3.11+, `uv`, en draai dezelfde commando's.
  De `video`-extra (mediapipe) werkt doorgaans goed op Windows.
- API-keys gaan in `.env` (nooit in code). Zie `.env.example`.

## Licentie

MIT — zie `LICENSE`.
