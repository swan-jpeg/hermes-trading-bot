# Plan: Twee-fase RL-training met historische gebeurtenis-replay

## Doel

Bouw Optie C: eerst synthetische RL-training, daarna fine-tune op ECHTE
historische gebeurtenis-data die "live" wordt afgespeeld alsof het toen
gebeurde. De hele keten (webscraping → impact agent → fusion → RL → risk →
Monte Carlo) moet end-to-end draaien.

## Architectuur

```
Historische gebeurtenis (GDELT, datum X)
    → Impact Agent (gratis Nous LLM) → impact-vector (categorie 1)
    → fusion (categorie 4) + bottleneck (categorie 5) + bron-info (categorie 2)
    → AssetContext (44-dim) → RL-model
    → risk engine + Monte Carlo → exposure
    → reward (return - drawdown - turnover)
```

De webscraping wordt GESIMULEERD alsof het live is: op elke historische dag
"verschijnt" de gebeurtenis van die dag, de impact agent vertaalt hem nu
(met de Nous-LLM), en de prijsreactie van die dag is de reward.

## Taken (in volgorde)

### 1. Historische gebeurtenis-downloader (GDELT)
- Nieuw bestand `hermes_bot/replay/gdelt.py`.
- `fetch_events(query, start, end, maxrecords)` → lijst van
  `{date, title, url, domain, source}` via de GDELT 2.0 doc API
  (`https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=artlist&format=json`).
- Gratis, open, GEEN API-key. Offline-safe (lege lijst bij netwerkfout).
- Query per sector/onderwerp (bv. "semiconductor", "inflation", "Trump",
  "government spending") zodat de impact agent relevante instrumenten vindt.

### 2. Live-replay simulator
- Nieuw bestand `hermes_bot/replay/simulator.py`.
- `ReplaySimulator`: neemt historische gebeurtenissen + prijzen (yfinance),
  en speelt ze dag-voor-dag af alsof het live is.
- Op elke dag: de gebeurtenissen van die dag worden door de impact agent
  (Nous-LLM) vertaald naar impact-vectoren → AssetContext.
- De prijs van die dag bepaalt de reward (return - drawdown - turnover).
- Dit is de "webscraping gesimuleerd alsof het toen was".

### 3. Prijzen koppelen (yfinance)
- `load_prices(ticker, start, end)` via yfinance (al in train_rl.py).
- Koppel elke gebeurtenis aan de prijs van dezelfde dag.

### 4. Impact agent met gratis Nous model
- Gebruik `LLMImpactAgent` met het beste GRATIS Nous-model:
  **`stepfun/step-3.7-flash:free`** (provider nous, base_url
  `https://inference-api.nousresearch.com/v1`). Dit is de sterkste gratis
  voor tool-gebruik/JSON-output. NIET deepseek-v4-flash (dat is betaald).
- De API-key komt uit `~/.hermes/auth.json` (providers.nous.agent_key) —
  NIET in de repo, NIET hardcoded. Lees via env-var of auth.json.
- De impact agent vertaalt elke historische gebeurtenis naar een impact-vector.

### 5. Regionale scores via overheidsbronnen + OSINT
- Gebruik GDELT-queries op overheids/overheids-onderwerpen voor regionale
  scores (bv. "government spending", "infrastructure", "unemployment").
- Optioneel: opensource OSINT-tools (maigret/holehe) voor entiteit-verrijking
  — maar houd het simpel: GDELT-overheidsbronnen zijn voldoende voor nu.

### 5b. Data-opslag (BELANGRIJK)
- Alle grotere data (GDELT-export, historische gebeurtenissen, replay-data,
  gedownloade prijzen) gaan op de EXTERNE SSD `/mnt/ssd/trading-bot-replay/`.
  NIET op de kleinere interne SSD, NIET in de GitHub-repo.
- Maak het pad configureerbaar via env-var `REPLAY_DATA_DIR` (default
  `/mnt/ssd/trading-bot-replay/`), zodat het op de Windows-PC van de gebruiker
  naar een lokale map kan wijzen.
- Zet GEEN grote data-bestanden in de repo — alleen code + een klein voorbeeld.

### 6. Twee-fase training
- Nieuw bestand `hermes_bot/train_rl_replay.py` (of breid train_rl.py uit).
- Fase 1: synthetische AssetContexts (bestaande `build_asset_contexts`).
- Fase 2: fine-tune op echte historische gebeurtenis-data via de replay.
- Gebruik stable-baselines3 PPO (draait op de Windows-PC van de gebruiker,
  NIET op deze server — `import stable_baselines3` hangt hier).

### 7. End-to-end test van de hele keten
- Test dat webscraping → impact → fusion → RL → risk → Monte Carlo allemaal
  draait met de replay-data.
- Monte Carlo: gebruik `MonteCarloEngineV2` (al in risk_v2/montecarlo.py).
- Risk engine: `RiskEngineV21` (al in risk_v2_1/__init__.py).

## Harde regels

- **Tests mogen NOOIT zware modellen laden** (transformers/torch/mediapipe =
  harde SIGILL op server-CPU). Mock ze.
- **`import stable_baselines3` hangt op de server** (cuda.bindings) — RL-training
  alleen op de Windows-PC van de gebruiker. Tests mogen SB3 NIET importeren.
- **Geen secrets in code.** Nous API-key via env-var of auth.json, nooit
  hardcoded.
- **Geen persoonlijke paden/namen** (repo is public). Geen `/home/olivier`,
  geen `obsidian-vault`, geen `@ehl`.
- **Nederlands in conversatie, Engels in repo-code/docs.**
- **Tests per taak.** Elke nieuwe module krijgt tests.
- **`uv run pytest` + `uv run ruff check hermes_bot tests`** moeten groen zijn
  vóór commit.
- **Commit + push** na elke afgeronde taak.

## Acceptatiecriteria

1. `hermes_bot/replay/gdelt.py` haalt historische gebeurtenissen op (GDELT).
2. `hermes_bot/replay/simulator.py` speelt ze dag-voor-dag af met de impact
   agent (Nous-LLM) → AssetContext.
3. De hele keten draait end-to-end (impact → fusion → RL → risk → MC).
4. Twee-fase training: synthetisch eerst, dan fine-tune op echte data.
5. Tests groen, ruff schoon, alles gecommit + gepusht.