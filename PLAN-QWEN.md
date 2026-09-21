# Hermes Trading Bot — Implementatieplan voor Qwen3 Coder

> **Doel:** dit document is de enige bron die Qwen3 Coder (of een andere codeer-agent)
> nodig heeft om de trading-bot te bouwen. Volg het **strikt in volgorde**. Elke taak
> eindigt met een test die groen moet zijn vóór je verder gaat.
>
> **Belangrijkste regel:** gebruik **bestaande opensource-modellen/bibliotheken** voor
> zware componenten (emotie, gezichtsuitdrukking, spraak, sentiment). Codeer **alleen
> zelf** wat geen goed opensource-alternatief heeft. Zie de tabel in sectie 2.

---

## 0. Werkomgeving & regels (LEES EERST)

### 0.1 Projectlocatie
- Repo: `~/trading-bot/`
- Package: `hermes_bot/` (Python 3.11+, `uv`, `ruff`, `pytest`)
- Kennisbron (architectuur): `~/obsidian-vault/TradingBot/` (13 genummerde noten)
- Web-interface draait op poort 9124 (`hermes_bot/webui.py`)

### 0.2 Harde regels voor de codeer-agent
1. **Backtest vóór geld, paper vóór live.** Nooit live-geld zonder bewezen backtest.
2. **RL voorstelt, Risk keurt goed.** Een RL-output gaat NOOIT rechtstreeks naar de markt; altijd door `RiskEngine.approve()`.
3. **Fail-closed:** als broker onbereikbaar is → géén order.
4. **Decision ≠ executie:** log intentie én echte fills.
5. **Verifieer elke order** met de broker; geloof geen impliciete succes-aanname.
6. **Gebruik bestaande modellen** waar die bestaan (sectie 2). Schrijf geen eigen CNN voor gezichtsuitdrukking als `deepface`/`py-feat` het al doet.
7. **Tests per taak:** 1–2 gedragstests (contract, geen change-detector). Draai `uv run pytest` en `uv run ruff check hermes_bot tests` — beide groen vóór commit.
8. **Commit per taak** met een duidelijke boodschap.
9. **Niet over-engineeren.** Kwaliteit over complexiteit. Als iets met 20 regels kan, geen 200.
10. **Geen secrets in code.** API-keys alleen in `.env` (via `get_secret` in `hermes_bot/config.py`).

### 0.3 Commando's
```bash
cd ~/trading-bot
uv sync --extra dev          # deps + dev-tools
uv run pytest                # tests
uv run ruff check hermes_bot tests
uv run python -m hermes_bot.demo   # end-to-end demo
```

---

## 1. Huidige staat (wat al werkt)

Deze onderdelen zijn **al geïmplementeerd en getest** (12 tests groen). **Niet opnieuw bouwen** — alleen uitbreiden waar een taak dat vraagt:

| Module | Status |
|---|---|
| `schemas.py` | `Position`, `ExitReason`, `Action`, `FusionSignal`, `RLRawDecision`, `RiskApproval`, `MonteCarloResult` — klaar |
| `portfolio/` | `PortfolioState` (posities met entry-prijs), `PortfolioAllocator` — klaar |
| `rl/` | `RulePolicy` (prijs-bewust: take-profit/stop-loss/trailing), `RLFusionModel`, `compute_reward` — klaar |
| `risk/` | `RiskEngine.approve()` met Monte Carlo-begrenzing + exit-altijd-goedkeuring — klaar |
| `simulation/` | `MonteCarloEngine` (bootstrap, VaR95, ES95, drawdown, crash-kans) — klaar |
| `execution/` | `PaperBroker`, `ExecutionEngine` (fail-closed) — klaar |
| `agents/` | `FundamentalAgent` (regelsgebaseerde scores) — klaar |
| `fusion/` | `WeightedFusion` — klaar |
| `expansions/` | `BottleneckAnalyzer`, `RegimeDetector` — klaar |
| `demo.py`, `webui.py` | end-to-end demo + web-interface — klaar |

**Wat nog NIET is gebouwd** (de taken hieronder): echte data-collectors, backtest-framework, NLP-sentiment, audio/video-analyse, RL-training, broker-integratie, B2B-data.

---

## 2. Model-keuzes: gebruik bestaande opensource (NIET zelf coderen)

> **Regel:** voor deze componenten gebruik je de genoemde bestaande bibliotheek/model.
> Je schrijft alleen de **integratie-wrapper** (data in → features uit), niet het model zelf.

| Component | Aanbevolen opensource | Waarom | Taak |
|---|---|---|---|
| **Spraak→tekst** | **OpenAI Whisper** (`openai-whisper` of `faster-whisper`) | De facto standaard, geen concurrent verslaat het | T3 |
| **Spraak-emotie** | **SpeechBrain** (`speechbrain` pretrained emotion) of **wav2vec2-finetune** | Getraind op emotion datasets | T3 |
| **Prosodie (pitch/volume/pauzes)** | **openSMILE** (`opensmile` Python, eGeMAPS feature set) | Standaard voor affectieve spraakfeatures | T3 |
| **Gezichtsuitdrukking** | **DeepFace** (`deepface`, emotion) of **Py-Feat** (`feat`, Detectorv2) | Getraind op FER-datasets, 478-pt mesh | T4 |
| **Pose / lichaamstaal** | **MediaPipe Pose** (`mediapipe`) | 33 landmarks, snel, robuust | T4 |
| **Financieel nieuws-sentiment** | **FinBERT** (`ProsusAI/finbert`) of **FinGPT** | Getraind op financiële teksten | T2 |
| **Algemeen nieuws-sentiment** | **DeBERTa** (`microsoft/deberta-v3-base` finetune) | Overklast FinBERT/RoBERTa op nieuws | T2 |
| **RL-algoritme** | **Stable-Baselines3** (PPO/SAC) + **Gymnasium** | Standaard RL-framework | T6 |
| **RL-trading-omgeving** | **FinRL** (`AI4Finance-Foundation/FinRL`) als referentie | Bewezen trading-env-ontwerp | T6 |
| **Marktdata** | **yfinance** (gratis) / **Alpaca** (paper) | Al in `data/market.py` | T1 |
| **Backtest** | **vectorbt** of **backtrader** (of eigen event-driven) | Bewezen backtest-frameworks | T5 |

**Als een model niet in de lijst staat** (bijv. B2B-bottleneck, regionale scores): dat is **custom** — codeer het zelf, maar houd het simpel en testbaar.

---

## 3. Taken (volgorde is verplicht)

### T1 — Data-laag: echte marktdata-collector + opslag
**Bestanden:** `hermes_bot/data/market.py` (uitbreiden), `hermes_bot/data/storage.py` (uitbreiden), `tests/test_data.py` (nieuw)

**Wat:** de `MarketDataCollector.collect()` (nu `NotImplementedError`) moet echte OHLCV ophalen via yfinance en opslaan in SQLite. `SQLiteStore` bestaat al.

**Eisen:**
- `collect()` haalt dagelijkse OHLCV op voor de universe uit `config/config.yaml`.
- Sla op via `SQLiteStore.insert_many()` (bestaat al).
- Idempotent: herhaald draaien geeft geen duplicaten (PRIMARY KEY symbol+timestamp).
- Test: mock yfinance → assert records correct + idempotent.

**Test (gedrag):**
```python
def test_market_collector_idempotent(tmp_path, monkeypatch):
    # mock yfinance.download om vaste df terug te geven
    # collect() 2x -> storage heeft geen duplicaten
```

---

### T2 — NLP-sentiment op nieuws/reports
**Bestanden:** `hermes_bot/signals/textual.py` (uitbreiden), `hermes_bot/signals/sentiment.py` (nieuw), `tests/test_sentiment.py` (nieuw)

**Wat:** een `SentimentAnalyzer` die nieuws/reports omzet naar een sentiment-score (-1..1) met confidence, met bronweging.

**Model:** gebruik **FinBERT** (`ProsusAI/finbert`) via `transformers` voor financieel nieuws; val terug op een simpele lexicon-score als het model niet beschikbaar is (offline-safe).

**Eisen:**
- `analyze(texts: list[str]) -> list[AlertSignal]` met `sentiment` en `credibility`.
- Bronweging: `credibility` per bron (uit config of default 0.5).
- Offline-fallback: als `transformers`/model niet geïnstalleerd is, gebruik een lexicon (positief/negatief woordenlijst) — zodat tests zonder model draaien.
- Test: gegeven bekende zinnen → sentiment-teken klopt.

---

### T3 — Audio-analyse (spraak→tekst, emotie, prosodie)
**Bestanden:** `hermes_bot/signals/audiovisual.py` (uitbreiden), `hermes_bot/signals/audio.py` (nieuw), `tests/test_audio.py` (nieuw)

**Wat:** een `AudioAnalyzer` die een audio-bestand omzet naar `SpeechSignal` met transcript, emotie-scores en prosodie.

**Modellen (gebruik bestaande):**
- Transcript: **faster-whisper** (of openai-whisper).
- Emotie: **SpeechBrain** pretrained emotion model.
- Prosodie: **openSMILE** (eGeMAPS) → pitch, volume, pauses.

**Eisen:**
- `analyze(audio_path) -> SpeechSignal` met `transcript`, `emotion_scores{confident,anxious,optimistic}`, `prosody{pace,pitch_var,volume}`, `pauses`.
- **Offline-safe:** als de modellen niet geïnstalleerd zijn, geef een `SpeechSignal` met lege/neutrale waarden + `confidence=0.0` (zodat de pipeline niet crasht).
- Test: mock de model-calls → assert `SpeechSignal`-structuur.

---

### T4 — Video-analyse (gezichtsuitdrukking, pose, publiek)
**Bestanden:** `hermes_bot/signals/video.py` (nieuw), `tests/test_video.py` (nieuw)

**Wat:** een `VideoAnalyzer` die een video/frame omzet naar video-features.

**Modellen (gebruik bestaande):**
- Gezichtsuitdrukking: **DeepFace** (`deepface`, emotion) of **Py-Feat**.
- Pose/lichaamstaal: **MediaPipe Pose** (33 landmarks).
- Publiek: simpele frame-analyse (aantal gezichten via MediaPipe Face Detection).

**Eisen:**
- `analyze(video_path) -> dict` met `facial_expression`, `gesture`, `body_language`, `audience`.
- **Offline-safe:** zonder modellen → neutrale dict + `confidence=0.0`.
- Test: mock → assert structuur.

---

### T5 — Backtest-framework + benchmark
**Bestanden:** `hermes_bot/backtest/` (nieuw: `engine.py`, `metrics.py`), `tests/test_backtest.py` (nieuw)

**Wat:** een event-driven backtester die een strategie (bijv. de `RulePolicy` + `RiskEngine`-keten) over historische data draait en metrics berekent.

**Eisen:**
- `Backtester.run(prices, strategy) -> BacktestResult` met equity-curve.
- Metrics: totaal rendement, Sharpe, Sortino, max drawdown, win-rate.
- **Benchmark:** buy-and-hold als baseline (de bot moet dit verslaan).
- Test: gegeven een simpele stijgende reeks → buy-and-hold rendement klopt.

---

### T6 — RL-omgeving + training (INPUT/OUTPUT-SPEC — zie sectie 4)
**Bestanden:** `hermes_bot/rl/env.py` (nieuw), `hermes_bot/rl/train.py` (nieuw), `tests/test_rl_env.py` (nieuw)

**Wat:** een Gymnasium-omgeving die de `RulePolicy`-state als observation gebruikt, en een trainingsscript met Stable-Baselines3.

**Belangrijk:** de **exacte input/output-spec staat in sectie 4**. Bouw de omgeving volgens die spec. Het RL-model zelf (PPO/SAC) is Stable-Baselines3 — niet zelf schrijven.

**Eisen:**
- `TradingEnv` (Gymnasium) met observation = de state-vector uit sectie 4.
- Action space = `{BUY, SELL, HOLD}` met allocatie-gewicht.
- Reward = `compute_reward` (bestaat al).
- `train.py` traint PPO en slaat het model op.
- Test: env reset/step werkt, observation-shape klopt.

---

### T7 — Broker-integratie (paper → live)
**Bestanden:** `hermes_bot/execution/brokers/alpaca.py` (nieuw), `tests/test_alpaca.py` (nieuw)

**Wat:** een `AlpacaBroker` (paper) die de `BaseBroker`-interface implementeert.

**Eisen:**
- Implementeer `submit`, `get_positions`, `health` via Alpaca's REST API (paper).
- **Fail-closed:** als Alpaca onbereikbaar is → `health()` False → `ExecutionEngine` weigert.
- Test: mock Alpaca API → assert order-status + fail-closed.

---

### T8 — B2B-bottleneck + regionale scores (custom)
**Bestanden:** `hermes_bot/expansions/b2b.py` (uitbreiden), `hermes_bot/signals/regional.py` (nieuw), `tests/test_b2b.py` (nieuw)

**Wat:** de `BottleneckAnalyzer` (bestaat) koppelen aan echte data-bronnen, en regionale scores uit nieuws/alerts.

**Eisen:**
- `BottleneckAnalyzer` uitbreiden met een `ingest(reports)` die uit earnings/reports vraagsignalen haalt (orderboeken, lead times, capaciteit).
- `RegionalScorer` die nieuws/alerts per regio omzet naar de 4 scores (veiligheid, tevredenheid, sociale zekerheid, bedrijfseconomisch).
- Test: gegeven voorbeeld-reports → bottleneck-score + regionale scores.

---

## 4. RL-model: INPUT/OUTPUT-SPEC (bouw de omgeving hierop)

> **Als het RL-model zelf te ingewikkeld is om te trainen, is dit de volledige spec.**
> De omgeving (T6) moet deze exacte contracten implementeren. Het model (PPO/SAC)
> is Stable-Baselines3 — je hoeft geen eigen RL-algoritme te schrijven.

### 4.1 Observation (state-vector per entiteit)
De policy ziet per entiteit een genormaliseerde vector:

| Index | Feature | Bereik | Bron |
|---|---|---|---|
| 0 | sentiment (fusie) | -1..1 | `FusionSignal.emotie.sentiment` |
| 1 | zekerheid (fusie) | 0..1 | `FusionSignal.zekerheid` |
| 2 | kwaliteit (fusie) | 0..1 | `FusionSignal.kwaliteit` |
| 3 | regime | 0..3 (bull/bear/highvol/crash) | `RegimeDetector` |
| 4 | ongerealiseerde P&L % | -1..+5 | `Position.unrealized_pnl_pct` |
| 5 | holding-dagen | 0..N | `now - entry_time` |
| 6 | huidige allocatie | 0..1 | `PortfolioState` |
| 7 | VaR95 (portfolio) | -1..0 | `MonteCarloResult.var_95` |
| 8 | crash-kans | 0..1 | `MonteCarloResult.crash_probability` |

**Observation shape:** `Box(low=-inf, high=+inf, shape=(9,))`

### 4.2 Action space
`Discrete(3)` → `{0: HOLD, 1: BUY, 2: SELL}`. De allocatie-grootte wordt door de **RiskEngine** bepaald (niet door RL) — RL kiest alleen de richting. Dit houdt het simpel en veilig.

### 4.3 Reward
`compute_reward(returns, drawdowns, turnover, lam_dd=0.5, lam_to=0.02)` — bestaat al.
- Beloont rendement, straft drawdown (crashweerstand) en turnover (overtraden).

### 4.4 Verwachte waarden (kalibratie-richtlijn)
- **Bull-regime, positief sentiment, hoge zekerheid** → RL zou BUY moeten prefereren.
- **Take-profit bereikt (+15%)** → RL zou SELL moeten prefereren (winst nemen).
- **Stop-loss bereikt (-8%)** → RL MOET SELL (verlies beperken) — dit is een harde regel in `RulePolicy`, RL leert hetzelfde.
- **Crash-regime / hoge crash-kans** → RL zou HOLD/SELL moeten prefereren (risk-off).

### 4.5 Veiligheid
- RL-output gaat ALTIJD door `RiskEngine.approve()`.
- Exit-besluiten (take-profit/stop-loss) worden door RiskEngine **altijd goedgekeurd** (nooit geblokkeerd).

---

## 5. Acceptatiecriteria (alles moet groen)

- [ ] `uv run pytest` → alle tests slagen (bestaande 12 + nieuwe per taak).
- [ ] `uv run ruff check hermes_bot tests` → schoon.
- [ ] `uv run python -m hermes_bot.demo` → draait end-to-end (fusie → RL → risk → executie).
- [ ] Web-interface `http://localhost:9124` toont de architectuur + code + vault.
- [ ] Elke taak heeft 1–2 gedragstests (geen change-detectors).
- [ ] Geen secrets in code; alleen via `.env` + `get_secret`.

---

## 6. Volgorde & afhankelijkheden

```
T1 (data) ──► T2 (sentiment) ──► T3 (audio) ──► T4 (video)
                │
                └──────────────► T5 (backtest) ──► T6 (RL-env) ──► T7 (broker)
                │
                └──────────────► T8 (B2B/regionaal)
```

- **T1–T4** zijn onafhankelijk van elkaar (parallel mogelijk).
- **T5** heeft T1 nodig (data) en T2 (signalen).
- **T6** heeft T5 nodig (backtest voor de env) en de RL-spec (sectie 4).
- **T7** is onafhankelijk (broker).
- **T8** heeft T2 nodig (nieuws/alerts).

---

## 7. Klaar? Volgende stap
Als alle taken groen zijn, is de bot klaar voor **paper trading** (T7) en daarna pas live met harde risicogrenzen. Zie `~/obsidian-vault/TradingBot/13-Roadmap-Implementatie.md` voor de volledige roadmap.
