# Hermes Trading Bot

Modulaire AI-trading-bot (multi-asset) volgens de architectuur in
`obsidian-vault/TradingBot/`. Python-first. Doel: **goede winst** én
**weerstand tegen marktcrashes**.

> ⚠️ BACKTEST VÓÓR GELD, PAPER VÓÓR LIVE. Dit is een skeleton — de
> risicolaag en backtests zijn verplichte poorten vóór enige live order.

## Snelle gids
```bash
uv sync              # venv + deps
uv run python -m hermes_bot.cli --help
uv run pytest        # tests
```

## Mappenstructuur (spiegelt de architectuurlagen)
```
hermes_bot/
├── data/            # LAAG 1-2: collectors, storage, features
├── signals/         # LAAG 3: audio/video, reports, alerts, regionale scores
├── agents/          # LAAG 4: fundamentele AI-agent
├── fusion/          # LAAG 5: fusiemodel (kwaliteit/zekerheid/emotie)
├── rl/              # LAAG 6: RL-besluitvorming (env, reward, policy)
├── simulation/      # LAAG 6: Monte Carlo (bootstrap, scenario's, metrics)
├── risk/            # LAAG 6: risico-engine (sizing, drawdown, stress, hedging)
├── portfolio/       # allocatie over assetklassen (multi-asset)
├── execution/       # LAAG 7: output actie → broker (paper/live)
├── expansions/      # aanbevolen uitbreidingen (B2B bottleneck, regime, ...)
└── cli.py           # CLI-entry
```

## Kernontwerpregels
1. **RL voorstelt, Risk keurt goed** — nooit een RL-output rechtstreeks naar de markt.
2. **Fail-closed**: als broker onbereikbaar is → géén order.
3. **Decision ≠ executie**: log intentie én echte fills, terug naar training.
4. **Verifieer elke order** met de broker; geloof geen impliciete succes-aanname.

## Kennisbron
Volledige architectuur & uitwerkingen: `obsidian-vault/TradingBot/Home.md`
(+ 12 genummerde modulenoten + roadmap).
