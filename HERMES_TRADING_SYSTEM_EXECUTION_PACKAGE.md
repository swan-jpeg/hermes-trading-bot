# Hermes Trading System Execution Package

**Version:** 1.0  
**Prepared:** 2026-10-13 (Europe/Amsterdam)  
**Repository inspected:** `/home/olivier/trading-bot`, branch `master`, HEAD `70be28d`  
**Purpose:** Self-contained plan and delegation prompts for the standard Hermes CEO agent and its coder agent.

---

## 1. Purpose and operating rules

This is a build plan, not authorization to run the planned operations. It records verified repository facts, canonical interfaces, dependencies, complete coder prompts, and approval gates. Follow tasks in dependency order, one at a time. Do not implement the roadmap in one large assignment.

The application is a research prototype. The current pipeline is not an approved trading route. Keep live trading disabled. Default to deterministic local fixtures and mocked external services. A task prompt may not override the approval policy in this document.

Statuses in the registry describe planning readiness, not implementation completion. The initial registry is saved here and must be updated with evidence after each delegated task. The CEO must reread the repository and registry on resume because code may have changed since this package was prepared.

### Evidence labels

- **Verified:** directly observed in repository code, Git history, tests, or stored artifacts.
- **Repository-reported:** written in a report or README, not independently reproduced during this audit.
- **Recommendation:** architecture decision for future work, not a validated performance result.
- **Unresolved:** evidence is insufficient; the specified prerequisite task must resolve it before dependent implementation.

### Out of scope

The B2B supply-chain bottleneck scanner is explicitly excluded. Do not broaden this project to implement it. Existing B2B files may be left intact; do not treat them as the economic graph foundation.

---

## 2. Verified repository baseline

### 2.1 Repository and verification

- HEAD is `70be28d`, “Concept: Impact Agent catalog + RAG design (database schema, no code yet)”. The repository was clean when inspected. `53b4cd4` exists in Git history and introduced v2.1 risk code, experiments, and tests.
- There is no project documentation directory or repository-local `AGENTS.md` in the inspected tree. Existing project docs are root `README.md`, `PLAN-QWEN.md`, `PLAN-REPLAY.md`, and `impact_agent/` docs.
- `pyproject.toml` defines Python >=3.11, pytest, Ruff, optional ML/data/multimodal dependencies, and an `uv.lock`. No CI workflow was found.
- Safe test run: `./.venv/bin/pytest tests --ignore=tests/test_replay_simulator.py --ignore=tests/test_replay_simulator_simple.py` produced **139 passed, 5 failed**, 6 warnings. The failures were `test_control_panel.py` backtests requesting SPY through yfinance; DNS/network was unavailable. The two excluded replay-simulator modules also attempt yfinance downloads. No live trading, model training, or separate backtest was run. `uv run pytest` could not create a temporary file in the read-only uv cache; the workspace venv was used.
- This test run is a baseline, not a test of any changes in this document. No source implementation task was performed.

### 2.2 Existing components

| Component | Verified current behavior | Main paths | Limitation |
|---|---|---|---|
| Market data | yfinance daily OHLCV collector uses adjusted prices for a one-month window; separate helpers fetch backtest prices. | `hermes_bot/data/market.py`, `hermes_bot/backtest/__init__.py` | No comprehensive validation, point-in-time contract, source/availability timestamps, or robust missing-data policy. Market timestamps can be naive session dates relabeled UTC. |
| RSS | Speech, report and alert RSS adapters; demo records; per-feed broad exception handling. | `hermes_bot/data/scraper.py` | Records use ingestion time rather than preserving feed publication time; summaries are not full transcripts; retry/scheduling/dedup/provenance are incomplete. |
| Storage | SQLite `market_data` OHLCV table, primary key `(symbol,timestamp)`, `INSERT OR REPLACE`. | `hermes_bot/data/storage.py` | No migration history, source document store, decision ledger, model registry, or backup/restore evidence. |
| Signals | Optional Whisper, FinBERT, prosody, face and pose paths; keyword/text signals. | `hermes_bot/signals/` | Tests mostly cover missing-file and structural/neutral behavior; no labeled task evaluation. Emotion is not evidence of speaker intent. |
| Impact | Keyword-sector matcher plus optional remote LLM interpretation. | `hermes_bot/impact.py`, `impact_agent/skill/` | Entity list is heuristic; default direction/magnitude derive from sector match count, probability equals confidence; LLM response is weakly validated and exceptions silently fall back. |
| Embedding/RAG | Design proposal only, with catalog/RAG schema concepts. | `impact_agent/DESIGN-catalog-rag.md` | No embedding call, model metadata, vector index, retrieval service or evaluation. |
| Graph | Small B2B bottleneck graph code. | `hermes_bot/expansions/` | Excluded scanner; not a general, temporal economic knowledge graph. |
| Fusion/context | Weighted source fusion and handcrafted numeric `AssetContext`. | `hermes_bot/fusion/`, `hermes_bot/rl/context.py` | These numeric features are not semantic embeddings and are not calibrated predictions. |
| RL | Gymnasium environment, rule policy, PPO train script and inference wrapper. | `hermes_bot/rl/`, `hermes_bot/train_rl.py` | Training includes centered-window future leakage and synthetic signals; environment reward timing is misaligned. Training/evaluation governance is insufficient. |
| Risk | Legacy `RiskEngine`, v2 engine and current pipeline v2.1 engine coexist. | `hermes_bot/risk/`, `hermes_bot/risk_v2/`, `hermes_bot/risk_v2_1/` | Legacy per-asset cap exists. V2.1 does not consume `max_asset_weight`; default pipeline does not load YAML; return history is not populated by pipeline. |
| Monte Carlo | Four path families, copulas, stress scenarios and bootstrap intervals. | `hermes_bot/risk_v2/montecarlo.py` | Simulated distributions and outputs have not been independently calibrated for the decision use. |
| Backtest | Event loop invokes legacy risk engine, rebalances on same row close, reports metrics and synthetic stress. | `hermes_bot/backtest/` | Core path lacks transaction costs and fill uncertainty; close-based decision timing is optimistic; some UI tests require external price data. |
| Portfolio | Simple state and target/delta allocator. | `hermes_bot/portfolio/` | Not connected to the pipeline’s v2.1 exposure outputs. `drawdown` property is a TODO returning zero. |
| Execution | PaperBroker and generic ExecutionEngine; Alpaca-named stub. | `hermes_bot/execution/` | Generic engine accepts any raw order without risk approval; Alpaca class makes no API calls, says healthy, and returns fabricated positions. |
| Operations | Web UI and local experiment artifacts. | `hermes_bot/webui.py`, `var/` | No durable end-to-end decision/fill ledger, robust reconciliation, model promotion gates or incident recovery evidence. |

### 2.3 Confirmed defects to address

1. **V2.1 hard limits:** `RiskEngineV21` reads `max_exposure`, but does not read `max_asset_weight`; per-call clipping is not portfolio aggregation. The legacy `RiskEngine` does enforce the 0.10 default cap. The earlier 10% concern therefore applies to the legacy engine’s observed behavior, while the current v2.1 concern is the inverse: its configured per-asset limit is ignored.
2. **Risk feed:** pipeline construction is `Pipeline(config=None)` → `{}`; it does not load `config/config.yaml`. Pipeline updates risk drawdown only when an external caller invokes `record_equity`; it never calls v2.1 `record_return`. Volatility/correlation/emergency calculations in that path see no realized portfolio history.
3. **Pipeline decisions:** it constructs `Action.BUY` unconditionally, uses a sentiment sign to set an intent, then calls the stateful risk engine separately for each entity. V2.1 does not use `intent_to_alloc` in sizing except its exit special case. Per-entity approvals are order-dependent due to shared `_smooth` state. Pipeline never invokes `PortfolioAllocator` or `ExecutionEngine`.
4. **Order boundary:** ExecutionEngine has no approval argument or verification. The Alpaca adapter is an offline dummy and must not be represented as live/paper connectivity.
5. **Temporal leakage:** RL feature creation uses centered convolutions; environment computes the previous-to-current return after applying the current action. The backtest decides from the current close and rebalances at that close without transaction costs.
6. **Data times:** RSS collectors stamp `datetime.now()` instead of feed publication; daily market session dates are not a sufficient point-in-time timestamp contract.
7. **Evaluation:** the OOS manifest/report records a v2 experiment on SPY/QQQ/IWM from 2018-09-24 to 2026-09-21 and test window 2024-01-01 to 2026-09-21. These are repository-reported, not reproduced here; they do not validate v2.1 or prove profitability. The report describes some old-engine values as estimates. No raw price dataset was found alongside the listed OOS artifacts.

### 2.4 Open decisions already resolved for this plan

- **Start with a deterministic baseline, not RL.** RL remains disabled until causal features, realistic simulator timing/costs, and an independent holdout evaluation pass.
- **Initial persistence:** SQLite and versioned SQL migrations are sufficient for the tiny MVP. Keep raw text as local files with content hashes and database metadata. Do not add Postgres, hosted vector DB, or graph database before a measured need.
- **Graph MVP:** relational node/edge tables with typed temporal edges and evidence references. No learned graph embeddings in the MVP.
- **Vector MVP:** document retrieval only, behind an interface; use deterministic mock vectors in ordinary tests. Downloading or running a real pretrained model requires a separate approval-gated task. Candidate model details appear in section 5.8 and must be pinned and rechecked at execution time.
- **Broker MVP:** mock broker and paper simulation only. No real broker API task is enabled by this package.
- **Risk API:** approve the whole proposed target portfolio in a single call and produce an immutable approval object. Per-order boundary verifies that approval; LLM output never decides risk limits.

---

## 3. Target architecture and module boundaries

```text
Approved source adapters
  ├─ market bars / portfolio observations
  └─ news, filings, releases, transcripts
       ↓
Normalize + validate + provenance + point-in-time availability
       ↓
Canonical catalog + relational store + local raw document store
       ↓
Document chunks → optional versioned document retrieval
       ↓
Evidence-linked event/claim extraction
       ↓
Typed temporal economic graph + bounded retrieval
       ↓
Impact estimator (association/structure/causality kept distinct)
       ↓
Decision engine: deterministic baseline → target weights
       ↓
Independent portfolio risk gate → immutable approval
       ↓
Order manager → mock broker / paper simulator
       ↓
Reconciliation, decision ledger, metrics, evaluation and model governance
```

### Module responsibilities

1. **Adapters:** fetch only when explicitly enabled; retain raw source and source timestamps; return typed records. Adapters do not make trading decisions.
2. **Normalization:** validate units, timezone, identifiers, duplicate keys, source availability and stale-data policy. Invalid inputs are rejected or marked unavailable; never converted silently into valid neutral values.
3. **Catalog/store:** canonical stable IDs, aliases, instruments and temporal relationships; append-only evidence and decision records. Store selection remains simple until benchmarks justify change.
4. **Document retrieval:** retrieval only. Similarity supports candidate discovery and does not imply economic relationship, return correlation or causality.
5. **Impact estimator:** outputs affected canonical entities, event type, direction/horizon estimates and evidence references with calibrated or explicitly uncalibrated uncertainty. It does not output trade approval.
6. **Decision engine:** consumes versioned market/portfolio/impact features and proposes desired weights. The MVP is a deterministic baseline whose action timing and costs can be tested.
7. **Risk engine:** independently validates complete target portfolio, hard constraints and input freshness. Missing required data or malformed approval requests fail closed.
8. **Order manager:** accepts only a risk-approved target transition; derives bounded order intents; prevents duplicate submissions and reconciles broker state. Initial broker is local mock only.
9. **Evaluation/governance:** records dataset, code and model versions; runs reproducible chronological evaluation; requires explicit promotion before model changes influence a trading path.

---

## 4. Canonical contracts

All future tasks use these contracts. A coder must not invent alternate field names or units. Add Pydantic models under `hermes_bot/schemas.py` or a narrowly scoped submodule if that file becomes unwieldy; preserve existing public schemas and add explicit conversion adapters.

### 4.1 Time, identity, and units

- Timestamps are timezone-aware UTC ISO-8601 at API/storage boundaries. Keep `event_time`, `published_time`, `ingested_at`, `available_at`, and `decision_time` distinct when known. Unknown source time is `null` with a reason, not replaced by ingestion time.
- Bars have interval start/end and source; daily exchange sessions keep exchange/calendar identity and session date. Never label a date-only value as a precise UTC close.
- Persistent entities use opaque stable IDs `ent_<UUID>`; instruments use `ins_<UUID>`; events `evt_<UUID>`; documents `doc_<sha256-hex>` based on normalized source URI plus raw content hash; chunks `chk_<sha256-hex>` based on document ID, chunker version and offsets. UUIDs are generated once and stored; ticker is an alias, not an ID.
- Instrument fields distinguish `issuer_entity_id`, asset class, exchange MIC, currency, instrument type and validity interval. Do not conflate company and equity, bond and issuer, commodity and futures contract.
- Prices and cash are decimal amounts in named currency; returns are decimal fractions (`0.01` means 1%); portfolio weights/exposures are NAV fractions (`0.10` means 10%); quantity is instrument units; annualized volatility is decimal per year; timestamps and horizons are explicit.
- Confidence fields are floats in `[0,1]`, but represent a calibrated probability only when a named calibration method and evaluation set exist. Otherwise field semantics must say `model_score` or `subjective_confidence`.

### 4.2 Market bar

```json
{
  "instrument_id": "ins_<UUID>", "source_id": "string",
  "interval": "1d", "session_date": "YYYY-MM-DD", "exchange_mic": "XNAS",
  "interval_start": "UTC timestamp or null", "interval_end": "UTC timestamp or null",
  "available_at": "UTC timestamp", "currency": "USD",
  "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
  "volume": 1.0, "adjustment": "raw|split_adjusted|total_return_adjusted",
  "source_revision": "string or null", "quality_flags": []
}
```

Required invariants: finite values; positive prices; `low <= min(open,close) <= max(open,close) <= high`; volume non-negative; unique key `(instrument_id, source_id, interval, session_date, source_revision)` or documented provider equivalent. Reject conflicts instead of silently overwriting historical values.

### 4.3 Document, chunk, entity, relation, event

- **Document:** `document_id`, `source_id`, `source_uri`, `raw_sha256`, `language`, `document_type`, `published_time`, `ingested_at`, `available_at`, `license_id`, `raw_object_path`, `metadata`.
- **Chunk:** `chunk_id`, `document_id`, `chunker_version`, `start_offset`, `end_offset`, `text`, `section`, `speaker_id` optional, `time_start_ms/time_end_ms` optional. Keep source offsets exact.
- **Entity:** `entity_id`, `entity_type` (`company|security|bond|fund|index|commodity|contract|currency|country|government|central_bank|person|organization|indicator|industry`), canonical name, validity interval, metadata.
- **Alias:** `alias`, namespace/source, entity ID, validity interval, resolution status. Ambiguous aliases return candidate set; never silently pick a ticker match.
- **Relation edge:** `edge_id`, `from_entity_id`, `relation_type`, `to_entity_id`, `valid_from`, `valid_to`, `observed_at`, `available_at`, `evidence_ids[]`, `source_ids[]`, `strength_value` nullable, `strength_semantics` (`observed_association|structural_dependency|causal_estimate|unknown`), `confidence` nullable, `method_version`.
- **Event/claim:** `event_id`, event type, normalized claim text, stance/direction nullable, affected entity IDs, `event_time`, `published_time`, `available_at`, source document/chunk IDs, epistemic type (`confirmed_fact|allegation|forecast|opinion|unknown`), novelty nullable, uncertainty fields, extractor version. Event sentiment is not itself a price prediction.

### 4.4 Retrieval and embedding metadata

`EmbeddingRecord` contains `embedding_id`, `object_id`, `object_type`, `task` (`document_retrieval` initially; later distinct entity/event/graph task), `model_id`, immutable `model_revision`, `tokenizer_revision`, `dimension`, `preprocessing_version`, `chunker_version`, `normalized`, `similarity_metric`, `created_at`, vector checksum, and vector. A vector index is partitioned by `(task, model_id, model_revision, dimension)`; incompatible vectors may not share an index. Query response includes IDs, scores, metadata and source offsets. Scores are retrieval scores, not probabilities or economic relations.

### 4.5 Impact output

```json
{
  "event_id": "evt_<UUID>", "as_of": "UTC timestamp",
  "extractor_version": "string", "evidence_ids": ["chk_<sha256>"],
  "impacts": [{
    "entity_id": "ent_<UUID> or ins_<UUID>", "impact_type": "revenue|cost|supply|demand|policy|rates|other",
    "direction": "positive|negative|mixed|unknown", "horizon_days": [1, 5, 20],
    "estimated_return": null, "estimated_return_semantics": "none|association|causal",
    "subjective_confidence": 0.0, "calibration_id": null,
    "evidence_ids": ["chk_<sha256>"], "reason_codes": []
  }], "warnings": []
}
```

No impact output is a trade, a risk approval, or proof of causality. `estimated_return` remains null unless a separately evaluated prediction model explicitly supplies it.

### 4.6 Decision, risk approval, and order intent

- **Target position:** `instrument_id`, `target_weight` (NAV fraction, long-only MVP `[0,1]`), `decision_time`, `strategy_id/version`, `input_record_ids[]`, `reason_codes[]`.
- **Risk request:** complete target portfolio plus current holdings/cash/NAV, price/FX freshness, constraints version, `as_of`; one request per portfolio decision cycle.
- **Risk approval:** `approval_id`, request hash, `approved|reduced|rejected|paused`, full approved target map, constraint version, approved at, expires at, per-constraint checks and reason codes. Immutable. Invalid/stale request, unknown instrument, missing price/FX, non-finite value or bad configuration rejects/pauses.
- **Order intent:** `order_intent_id`, approval ID, instrument ID, side, quantity, order type, limit price optional, currency, time-in-force, idempotency key, created time, expiry. Quantity and price precision comes from instrument metadata. Initial mock supports market and limit intent only; no shorts/margin.
- **Broker result:** client/idempotency ID, broker order ID, status (`accepted|partially_filled|filled|rejected|cancelled|unknown`), fills with qty/price/time/fees, error code. Timeout state is `unknown` until reconciled; never blindly resubmit.

### 4.7 Errors and configuration

- Typed errors: `ValidationError` (bad input), `DataUnavailable` (required source absent/stale), `ApprovalRejected` (with reason codes), `ExternalServiceError`, `DuplicateRequest`, `ReconciliationRequired`. Never swallow errors that affect safety; fallback output includes an explicit degraded-state flag.
- Start from `config/config.yaml` through `load_config`, but extend it with validated typed settings. Safe defaults: `execution.mode=mock`, `execution.live_enabled=false`, `data.mode=OFFLINE_MOCK`, no network adapter active, `risk.fail_closed=true`, zero leverage, no shorting. A flag alone is not a hard security boundary; live adapter must not exist in the MVP.
- Config must state limits explicitly: `risk.max_asset_weight`, `risk.max_gross_exposure`, `risk.max_sector_weight`, `risk.max_leverage=0`, `risk.min_cash_weight`, `risk.max_stale_seconds_by_source`, `risk.max_position_count`. Missing required limits fail validation. For `OFFLINE_MOCK`, freshness is determined only by fixture `available_at <= as_of`; no age threshold is inferred. An enabled external source requires an explicit age threshold before it can be used for a decision.
- Initial long-only MVP limits are fixed as follows unless the user explicitly changes scope: `max_asset_weight=0.10` (existing YAML value), `max_gross_exposure=1.00`, `max_sector_weight=0.20` (existing `max_sector_risk` converted to a weight only after sector IDs are available), `max_leverage=0.0`, `min_cash_weight=0.0` in normal state, `cash_min_in_crash=0.30` (existing YAML value), and `max_position_count=8` (the eight configured stock/ETF/bond symbols). No inferred sector cap is applied when sector identity is unresolved; the unknown-sector position still obeys per-asset and gross caps. Stale-data age is mandatory configuration per source; the offline MVP uses fixture `available_at` ordering and does not pretend synthetic age thresholds validate a real market feed.

### 4.8 Embedding recommendation

No model is installed or used by this repository. For a later English document-retrieval experiment, benchmark two frozen candidates on the project’s own labeled queries: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional sentence vectors) and `BAAI/bge-small-en-v1.5` (384 dimensions, model card lists 512 sequence length). These are candidates only, not validated choices. Confirm current model card, license, exact immutable revision, tokenizer, pooling, maximum effective input length, and offline weight availability before any download. Model-card dimensions/limits are not repository facts. Prefer fixed version and cosine similarity on L2-normalized vectors; test exact preprocessing and query instruction behavior per model. Use a local exact-search implementation for tiny corpus first; no vector database dependency until benchmarked. Document retrieval only in MVP. No entity/event/graph embeddings or fine-tuning in initial release. (Model references: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE small v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5).)

---

## 5. Dependency graph and roadmap

```text
T01 risk contract/config + hard approval
 ├─→ T02 pipeline risk-state wiring
 ├─→ T09 decision integration
 └─→ T10 mock order manager
T03 causal simulator/backtest ─→ T09 baseline decision evaluation
T04 catalog/provenance/schema ─→ T05 offline ingestion ─→ T06 document retrieval
                         └────→ T07 temporal economic graph
T04 + T05 + T07 ──────────────→ T08 deterministic evidence-linked Impact Agent
T03 + T08 + T09 ──────────────→ T11 point-in-time impact evaluation
T01 + T02 + T09 + T10 ────────→ T12 ledger/monitoring/recovery
T06 real model, external data adapters, real brokers, model training and scale work
    require separate explicit approval and are not auto-enabled by this package.
```

### Phases

| Phase | Tasks | Exit condition |
|---|---|---|
| 0. Safe baseline and contracts | T01 | Risk inputs, constraints and approval are explicit and tested offline. |
| 1. Connect deterministic state | T02 | Pipeline receives validated config, price state and realized returns; fail-closed. |
| 2. Simulation correctness | T03 | Causal feature/action timing and net-of-cost backtest fixture pass. |
| 3. Catalog and source provenance | T04, T05 | Stable IDs, timestamped records and tiny synthetic storage/replay work. |
| 4. Retrieval and economic relations | T06, T07 | Mock retrieval and temporal graph contracts pass; no real model/data required. |
| 5. First Impact Agent | T08 | Deterministic extraction grounded in evidence; uncalibrated fields labeled. |
| 6. Baseline portfolio decision | T09 | One deterministic strategy proposes complete target weights after costs/constraints. |
| 7. Evaluation | T11 | Chronological impact evaluation produces reproducible metrics and limitations. |
| 8. Mock execution and operations | T10, T12 | Approval-gated mock order lifecycle, reconciliation, ledger and recovery tests pass. |
| 9. Approved optional research | T13+ | External feeds, model weights, supervised/RL training, live broker or scaling only by separately approved scoped task. |

---

## 6. Task registry

Initial statuses are `READY` only for tasks that need no external approval and whose prerequisites are satisfied. All others are `BACKLOG`; external operations are `NEEDS_APPROVAL`. After each task, update status and evidence in this table before proceeding.

Allowed states: `BACKLOG`, `READY`, `IN_PROGRESS`, `REVIEW`, `VERIFIED`, `BLOCKED`, `NEEDS_APPROVAL`, `FAILED`.

| ID | Task | Priority | Dependencies | Initial status | Prompt | Expected files | Required evidence / approval |
|---|---|---:|---|---|---|---|---|
| T01 | Portfolio-level v2.1 risk contract and hard limits | P0 | None | READY | T01 | `schemas.py`, `risk_v2_1/`, risk tests | Boundary and aggregate tests; no external approval |
| T02 | Validated config and pipeline risk-state feed | P0 | T01 | BACKLOG | T02 | `config.py`, `pipeline.py`, tests | Integration evidence showing risk history changes |
| T03 | Causal RL environment and cost-aware backtest timing | P0 | None | READY | T03 | `rl/env.py`, `train_rl.py`, `backtest/`, tests | Synthetic timestamp/price fixtures; offline only |
| T04 | Canonical catalog, provenance models and SQLite migrations | P1 | T01 contracts | BACKLOG | T04 | `schemas.py`, new `data/` migration/store modules, tests | Fresh temp DB schema + tiny fixture migration |
| T05 | Offline-first ingestion interfaces and timestamp normalization | P1 | T04 | BACKLOG | T05 | `data/`, source adapter tests | Mock RSS/market records only; network disabled |
| T06 | Document chunking/retrieval interface and mock vector index | P2 | T04, T05 | BACKLOG | T06 | new `retrieval/`, tests | Deterministic mock vectors; real model explicitly excluded |
| T07 | Typed temporal economic graph store and bounded traversal | P2 | T04 | BACKLOG | T07 | new graph module/schema/tests | 10-node synthetic graph; B2B scanner excluded |
| T08 | Deterministic evidence-linked event/impact extraction | P1 | T04, T05, T07 | BACKLOG | T08 | `impact.py` or new `impact/`, tests | Synthetic documents/events only; no external LLM |
| T09 | Decision engine and deterministic baseline target allocator | P1 | T01, T02, T03, T08 | BACKLOG | T09 | `pipeline.py`, `portfolio/`, `rl/`, tests | No model training; whole-portfolio risk contract used |
| T10 | Approval-bound mock order manager and reconciliation | P1 | T01, T09 | BACKLOG | T10 | `execution/`, schemas, tests | Local mock only; direct orders without approval rejected |
| T11 | Point-in-time impact and strategy evaluation harness | P2 | T03, T04, T08, T09 | BACKLOG | T11 | new evaluation module, tests, docs | Small synthetic labeled corpus; no performance claims |
| T12 | Decision ledger, health metrics and recovery protocol | P2 | T02, T09, T10 | BACKLOG | T12 | `data/`, pipeline, monitoring/tests | Local temporary store only; no production DB |
| T13 | Optional pretrained embedding model integration | P2 | T06; user approval | NEEDS_APPROVAL | T13 | retrieval module, tests, model metadata | Requires per-operation model download approval; no bulk embedding |
| T14 | Optional real source connector pilot | P2 | T05; user approval | NEEDS_APPROVAL | T14 | one adapter only | Exact source/request/record limit approval required |
| T15 | Optional supervised or RL research study | P3 | T03, T09, T11; user approval | NEEDS_APPROVAL | training/evaluation modules | Training/data/cost/runtime approval; no promotion or live use |
| T16 | Optional real broker integration | P0 if ever pursued | T01, T09, T10, T12; separate approval | NEEDS_APPROVAL | broker adapter only | Explicit paper-account scope approval; never live by default |
| T17 | Benchmark-driven scale review | P3 | T06, T07, T12; measurements | BACKLOG | benchmark docs/scripts | Small local benchmarks first; no infrastructure purchase/action |

---

## 7. Complete coder-agent prompts

Each block below is a complete prompt. The CEO must pass exactly one prompt at a time. Preserve prompt ID/version in the registry. If code has diverged, stop and update this package/task prompt before delegation.

### T01 — Portfolio-level v2.1 risk contract and hard limits

```text
Task ID/version: T01 / v1
Title: Portfolio-level v2.1 risk contract and hard limits
Phase/priority: Phase 0, P0
Dependencies: None. Approval: no external/network/data approval; local synthetic tests only.

Repository context: Current engine is hermes_bot/risk_v2_1/__init__.py. It returns one exposure per approve() call, reads max_exposure but not max_asset_weight, and stores prev_exposure for smoothing. hermes_bot/risk/__init__.py is a legacy engine that does cap individual allocations at max_asset_weight. Pipeline calls v2.1 once per entity, which is order-dependent and not an aggregate portfolio check. Preserve existing imports and V21Breakdown where feasible. Related tests: tests/test_risk_v2_1.py, tests/test_core.py.

Objective: Add a deterministic whole-portfolio v2.1 approval API implementing contracts section 4.6. It must enforce configured hard per-asset and aggregate constraints on a complete proposed target map. Keep the legacy engine intact. Do not connect execution in this task.

Implementation specification:
- Add typed request/result schemas (existing schemas.py preferred). Request contains request_id, as_of, nav, cash_weight, current_weights, proposed target positions, required price freshness flags and constraints/config version. Result contains approval_id, request hash, status approved/reduced/rejected/paused, complete approved target map, checks, reason codes, created/expiry timestamps and constraints version.
- `RiskEngineV21.approve_portfolio(request)` is the canonical new entry point. No per-asset loop and no dependence on entity ordering. Existing `approve()` remains as a compatibility adapter marked legacy; it must not claim aggregate safety.
- Config names/units: max_asset_weight, max_gross_exposure, max_sector_weight (only if sector metadata provided), max_leverage=0, min_cash_weight, max_position_count. Weights are NAV fractions. Missing required values, NaN/inf, negative NAV, unknown ID, stale/missing required price, invalid timestamps, or unsupported short/margin request returns rejected or paused with explicit reason; never silently defaults a hard limit.
- Long-only MVP. Risk checks proposal map in deterministic sorted-ID order, but output invariance must not depend on sorting because logic is portfolio-level. Reductions must preserve configured cash/gross constraints; use proportional scale-down of all positive proposed weights when aggregate gross exceeds cap, then individual cap, then recheck aggregate. If constraints cannot be satisfied, reject.
- Approval object is immutable (frozen model or equivalent) and hashes canonical serialized request including config version. Expiry is configurable and deterministic in tests via injected clock or fixed `as_of`.
- Logging is not a substitute for returned reason codes. Do not log secrets.

Ordered steps: 1) inspect schemas/engine/tests; 2) write test cases first; 3) implement models and validator; 4) implement whole-portfolio checks; 5) retain compatibility adapter; 6) run focused and offline test suites; 7) report any changed contract.

Tests: valid under limits; single-name cap; aggregate gross cap; minimum cash; zero leverage/short rejection; max positions; NaN/inf/negative weights; missing/stale price; missing hard config; deterministic repeated result; permuted proposed mapping yields equal approval; approved weights sum within all constraints; compatibility legacy tests remain valid. Use `./.venv/bin/pytest tests/test_risk_v2_1.py tests/test_core.py` and broader offline selection only if safe. No network.

Acceptance: Every emitted approved target map satisfies all configured hard constraints; failed inputs do not yield an executable approval; output is order-independent; legacy tests pass; no external side effects.

Explicit exclusions: Do not wire pipeline/execution, change strategy parameters, modify data source, connect broker, use live prices, or claim strategy performance.

Resource budget: At most 100 synthetic positions in unit tests, no downloads, no DB writes beyond pytest temp directory, under 30 seconds and 1 GB RAM.

Stop conditions: Stop if schema compatibility requires a breaking change, any hard-limit semantics are ambiguous, or existing code implies leverage/shorting must be supported. Do not invent permissive defaults; report blocker.

Completion report: List files changed, API/schema details, tests and exact outcomes, constraint cases, remaining gaps, deviations, and git diff summary. Do not mark verified without inspecting results.
```

### T02 — Validated config and pipeline risk-state feed

```text
Task ID/version: T02 / v1
Title: Validated configuration and pipeline risk-state feed
Phase/priority: Phase 1, P0
Dependencies: T01 verified. Approval: local synthetic inputs only; no network.

Context: hermes_bot/config.py loads YAML but Pipeline in hermes_bot/pipeline.py defaults to `{}` and does not load it. Pipeline.record_equity updates drawdown only; it never calls RiskEngineV21.record_return. Default closes provider can call yfinance. Keep test path offline and preserve Pipeline(config=...) callers.

Objective: Make configuration explicit and validated at pipeline construction, and provide one clear method to update portfolio equity/return history exactly once per completed observation. Fail closed when required risk configuration or required prices are unavailable.

Specification: Add typed validation for risk limits from T01 and write the initial values in section 4.7 into `config/config.yaml` using the canonical names; preserve deprecated aliases only with explicit conversion. `Pipeline(config=None)` loads repository config via load_config; tests inject a complete config and deterministic closes_provider. Add `record_portfolio_mark(as_of, equity, return_decimal, source_quality)`; validate finite positive equity and finite return, append return once, update drawdown, prevent duplicate/out-of-order as_of marks, and expose a typed error on invalid sequence. Keep `record_equity` as deprecated adapter if compatibility requires it; do not infer returns from absent data. For OFFLINE_MOCK require only fixture available_at <= as_of; for any enabled external source require configured max age or mark data unavailable.

Steps: inspect config, current tests and pipeline; implement validation; add mark state and dedupe; thread validated config into v2.1; ensure tests inject offline provider; update docs/example config only as needed; run focused tests.

Tests: config loaded and override works; missing required risk limits rejected; valid UTC mark appends once; duplicate/out-of-order mark rejected; invalid equity/return rejected; empty return history remains explicitly unavailable; stale/missing close does not become valid neutral data; no network from default tests. Commands: `./.venv/bin/pytest tests/test_pipeline.py tests/test_risk_v2_1.py`.

Acceptance: Effective risk settings are inspectable and identical to validated config; integrated mark update populates risk state once, while data absence is fail-closed; all tests offline.

Exclusions: No decision engine replacement, broker, external feeds, risk tuning, or database implementation.

Budget: synthetic data only, at most 100 marks, under 30 sec/1GB; no external requests or persistent writes.

Stop if current callers rely on silent YAML defaults or duplicate mark semantics cannot be inferred safely. Report required migration rather than guessing.

Report files, settings contract, tests/results, and limitations.
```

### T03 — Causal RL environment and cost-aware backtest timing

```text
Task ID/version: T03 / v1
Title: Causal features and next-period execution in simulator
Phase/priority: Phase 1, P0
Dependencies: None. Approval: synthetic fixtures only.

Context: hermes_bot/train_rl.py builds momentum with `np.convolve(..., mode="same")`, which uses values on both sides of an index. hermes_bot/rl/env.py applies action at current index and then rewards the previous-to-current return. Backtester in hermes_bot/backtest/__init__.py sees current row close and trades at that close; costs are not deducted. tests/test_rl_env.py and tests/test_backtest.py cover existing behavior.

Objective: Remove confirmed temporal leakage and make simulated execution chronology explicit. Keep public environment/backtest interfaces compatible or provide a documented adapter.

Specification: Replace centered features with trailing-only computations; at index t, no field may depend on price/return after t. Define action at close t and apply exposure to t→t+1 return (or explicitly documented next-bar fill). Do not apply an action to a return already realized. Add configurable commission_bps and slippage_bps to backtest config, default 0 only for backward compatibility in tests; clearly label gross/net. Core evaluation configuration must set explicit values. Use next bar open if available; if only close data, use next close and document approximation. No same-close fill after observing same close. Preserve deterministic seed behavior.

Steps: inspect state indexing; write future-perturbation test; shift observations/rewards; add cost calculation and result fields; update backtest timing; update docs; run offline fixtures.

Tests: changing all prices after t cannot change observation at t; action at t cannot earn t-1→t return; known two/three-bar hand calculation; buy/sell fees and slippage reduce return; zero-cost compatibility; empty/one-row/invalid price input; synthetic crash. Use `./.venv/bin/pytest tests/test_rl_env.py tests/test_backtest.py tests/test_mc_wiring.py` only after confirming these commands do not fetch network; do not run control-panel live-data tests.

Acceptance: causal invariants are tested; fills occur after decision availability; net return reconciles exactly to hand calculation; deterministic offline suite passes.

Exclusions: No RL training, parameter tuning, external yfinance, new strategy, pretrained model, or profitability claims.

Budget: <=2,000 synthetic bars per test, <=30 sec, <=1GB RAM, no downloads/DB writes.

Stop if historical API semantics cannot be changed compatibly without widespread callers; propose adapter and stop before unrelated refactor.

Report files, indexing convention, cost units, commands/results, and unresolved simulator assumptions.
```

### T04 — Canonical catalog, provenance models and SQLite migrations

```text
Task ID/version: T04 / v1
Title: Canonical entity catalog and versioned local schema
Phase/priority: Phase 2, P1
Dependencies: T01 schemas available. Approval: local temporary SQLite only.

Context: hermes_bot/data/storage.py creates one market_data table at runtime. No migration framework or canonical identifiers exists. Existing records use ticker symbol and timestamp. Existing public SQLiteStore may be imported by tests.

Objective: Add the minimum canonical entity, instrument, alias, source, document metadata, event, relation, and migration foundations defined in section 4 without introducing a new database service.

Specification: Add migration files/table with monotonically increasing version and transactional apply; retain legacy market_data table and provide non-destructive migration. Add tables/entities for `entity`, `instrument`, `alias`, `source_record`, `document`, `event`, `relation_edge`, `decision_record` (decision table can be schema-only, populated in T12). Foreign keys enabled. Use stable opaque `ent_`, `ins_`, `evt_`, `doc_` formats and UTC timestamps. Store decimals with documented representation and currency. Unique constraints protect source dedup keys. Relation edges have temporal bounds, evidence IDs and semantics. Never rewrite existing rows silently.

Steps: inspect store and tests; implement migration runner; implement typed repositories and validation; add legacy compatibility; test fresh DB and upgrade from legacy DB in `tmp_path`; document migration rollback limits.

Tests: fresh schema; idempotent migration; upgrade preserves two synthetic bars; invalid IDs/temporal intervals rejected; FK and uniqueness enforced; relation evidence references; close removes connection; failed migration rolls back. `./.venv/bin/pytest tests/test_data.py` plus new tests.

Acceptance: all schema changes occur only in temporary test DBs; fresh and legacy schema paths pass; no production database path is opened or migrated.

Exclusions: No Postgres, external DB, vector index, bulk data import, broker, or B2B scanner.

Budget: <=10 synthetic entities, 10 instruments, 5 documents, 10 edges, 100 bars; <30 sec/1GB.

Stop if repository has an untracked/unknown DB format or migration could alter real data; do not inspect secrets or run on default var DB.

Report migration IDs, tables, compatibility behavior, tests, and any schema deviation.
```

### T05 — Offline-first ingestion interfaces and timestamp normalization

```text
Task ID/version: T05 / v1
Title: Typed offline ingestion adapters and timestamp provenance
Phase/priority: Phase 3, P1
Dependencies: T04 verified. Approval: no network; mock providers only.

Context: hermes_bot/data/market.py uses yfinance and turns daily date into UTC; scraper.py stamps fetched RSS records with current time. Existing collector interfaces are in hermes_bot/data/__init__.py and tests/test_data.py/tests/test_web_scraper.py. Network is unavailable in the test environment.

Objective: Normalize adapter outputs into canonical records while separating source time and ingestion time. Default execution remains OFFLINE_MOCK.

Specification: Add an adapter protocol with `fetch(request)->list[RawRecord]` and a normalizer returning MarketBar/Document metadata contracts in section 4. Adapters receive explicit `mode`, max_records, timeout and source config. `OFFLINE_MOCK` is default and must not instantiate network clients. RSS parser retains provider publication field and URI; invalid/missing published time stays null and gets quality flag. Market daily session date remains a date, not fake UTC timestamp. Idempotency key uses source ID + stable provider ID or content hash. Retry only transient errors with bounded attempts; errors surface as typed unavailable status. Licensing/source ID metadata required for records.

Steps: inspect existing collector contracts; add protocol and mock adapter; implement timestamp parser tests; adapt current data conversion methods without enabling URLs; isolate legacy yfinance adapter behind explicit NETWORK mode; document disabled status.

Tests: mock RSS with RFC date and malformed date; mock market date; duplicate same record; changed source revision; empty response; adapter exception; record limit; assert default mode never calls urlopen/yfinance. Use existing data/webscraper tests plus new tests.

Acceptance: OFFLINE_MOCK has zero network calls; all source and ingestion times remain distinct; duplicates are deterministic; errors are explicit.

Exclusions: Do not activate live RSS/yfinance, scrape broadly, store actual news, or modify production DB.

Budget: <=5 synthetic documents, <=100 bars, <=1 network client object (must remain unused), <30 sec/1GB.

Stop if source timestamp parsing requires assumptions about a publisher’s timezone; retain unknown and report it.

Report changed adapters, timestamp rules, tests, and proof default mode made no network call.
```

### T06 — Document chunking and mock retrieval

```text
Task ID/version: T06 / v1
Title: Versioned document chunking and retrieval interface with mock vectors
Phase/priority: Phase 4, P2
Dependencies: T04 and T05 verified. Approval: deterministic local tests only.

Context: There is no embedding dependency or vector database. `impact_agent/DESIGN-catalog-rag.md` is a proposal. Contracts for document/chunk/EmbeddingRecord are section 4.4. Candidate 384-dimensional models are not approved for download.

Objective: Implement chunk metadata and a provider-neutral retrieval interface using deterministic mock vectors only, so future model integration has a stable contract.

Specification: New `hermes_bot/retrieval/` package. Chunker uses paragraph boundaries then sentence boundaries; maximum 400 whitespace-token approximation per chunk, 40-token overlap; retain exact character offsets, section/source IDs and version `paragraph_sentence_v1`. Do not split Unicode code points. `EmbeddingProvider.embed(texts, metadata)->list[EmbeddingRecord]`; validate fixed dimension per provider metadata, finite values and vector count. `VectorIndex.upsert/search` is in-memory exact cosine search for tiny corpora only; normalize vectors; deterministic lexical-hash test provider is explicitly marked `test_only` and cannot be selected in application config. Retrieval filters by availability time and metadata and returns scores with source offsets. No semantic/economic interpretation.

Steps: implement chunker; define provider/index protocols and typed records; add test-only deterministic provider; implement in-memory exact index; test compatibility/version rejection; document limitations.

Tests: empty/long/unicode document; overlap and offsets; deterministic mocks; zero vector rejection; dimension/model-version mismatch; availability filter; ranking exact calculation; duplicate upsert. `./.venv/bin/pytest` focused new retrieval tests.

Acceptance: interfaces meet section 4.4; no package download, external call or persistent vector DB; every result maps to source text offsets.

Exclusions: No real model, embeddings for real corpora, remote API, vector DB, graph embeddings, or prediction use.

Budget: max 5 synthetic documents, 20 chunks, 20 vectors, 384 dimensions, <10 sec/512MB.

Stop if existing package conventions conflict or a real model is needed for any test. Do not download it.

Report files, chunker behavior, interface and tests.
```

### T07 — Typed temporal economic graph

```text
Task ID/version: T07 / v1
Title: Typed temporal economic relation store and bounded traversal
Phase/priority: Phase 4, P2
Dependencies: T04 verified. Approval: local synthetic fixture only.

Context: Current expansions graph is B2B scanner code and is excluded. No general economic graph exists. Canonical edge fields are section 4.3. Persistence is SQLite only.

Objective: Add a small typed relation repository and deterministic evidence-bounded traversal over canonical entities.

Specification: Add `economic_relation` types: `issuer_of`, `subsidiary_of`, `located_in`, `member_of_sector`, `supplies`, `customer_of`, `exposed_to`, `policy_applies_to`, `indicator_measures`, `currency_of`. Store edges in T04 relation table with effective and available times, evidence IDs, strength semantics. Queries require `as_of` and max_hops (hard cap 3), direction and allowed relation types; exclude edges unavailable at as_of or outside validity. Return path and all evidence IDs. No inference that semantic match creates edge. Unknown strength is null; confidence is nullable unless defined.

Steps: inspect T04 model; add enum validation and repository; implement temporal traversal with cycle prevention and max-hop hard limit; add tests; document query complexity.

Tests: typed allowed/unknown type; temporal valid/expired/future edge; cycle; path evidence; max hops 0..3 and reject >3; missing entity; deterministic ordering. Use 10 nodes/10 edges in temp DB.

Acceptance: results are bounded, point-in-time and evidence-linked; no B2B scanner integration; no edge produced without explicit input.

Exclusions: No graph database, graph embeddings, automatic relation extraction, real supplier data or causal claims.

Budget: <=10 nodes, <=10 edges, <=100 traversal visits per test, <10 sec/512MB.

Stop if T04 lacks relation evidence constraints or time semantics; report dependency issue.

Report schema/query rules, tests and complexity.
```

### T08 — Deterministic evidence-linked event/impact extraction

```text
Task ID/version: T08 / v1
Title: First deterministic evidence-linked Impact Agent
Phase/priority: Phase 5, P1
Dependencies: T04, T05 and T07 verified. Approval: synthetic data only; external LLM disabled.

Context: Current hermes_bot/impact.py matches keyword substrings to hard-coded tickers and manufactures directional fields. LLMImpactAgent sends external HTTP when a key is present and swallows errors. New canonical entity/event/impact contract is in section 4.5.

Objective: Build deterministic event/claim normalization and entity candidate resolution grounded in supplied text spans and catalog aliases. Do not claim return prediction.

Specification: Do not create a broad natural-language inference system in this task. Define a rule table schema with `rule_id`, exact case-insensitive trigger phrase, required/forbidden phrase lists, allowed relation types, and optional explicit direction (default unknown). Rules are injected as config/test fixture; no new finance rules or ticker mappings may be invented. A match returns only entities reached through T07 typed relations or exact catalog aliases, with the precise text span and relation evidence. If alias resolution is ambiguous, emit candidates and no resolved impact. Output event and impact contracts; evidence IDs must point to chunks and span offsets. Direction can be unknown/mixed; estimated_return is always null in this task. Confidence is subjective and explicitly labeled. Conflicting evidence returns warning and both evidence IDs. Replace silent fallback with explicit degraded/error state in existing adapter. Keep existing ImpactAgent public methods through compatibility adapter if tests or callers need them. Never send network.

Steps: inspect existing tests/API; define extractor protocol; implement deterministic fixture rule set; add catalog resolution/evidence; adapt old API without fabricated probability; audit no remote call in default mode; add tests.

Tests: known phrase maps to correct canonical ID and span; ambiguous alias returns candidates; unrelated text no impact; negation/conflict marked; allegation vs forecast preserved; source unavailable errors; confidence not called probability; malicious/invalid model-shaped values rejected. Only supplied synthetic text.

Acceptance: each non-null impact has traceable evidence; no ungrounded ticker; estimated return null; all network paths disabled by default; legacy tests are adapted or pass.

Exclusions: No LLM/network, real company research, sentiment-to-price conversion, trading decision, or B2B scanner.

Budget: 5 synthetic docs/events, 10 entities, <30 sec/1GB.

Stop if required event taxonomy or entity alias cannot be supported by current catalog; create a blocker report instead of guessing.

Report exact taxonomy/rules, APIs, tests, and unsupported cases.
```

### T09 — Decision engine and deterministic baseline target allocator

```text
Task ID/version: T09 / v1
Title: Replace hard-coded pipeline BUY with deterministic target-weight baseline
Phase/priority: Phase 6, P1
Dependencies: T01, T02, T03, T08 verified. Approval: synthetic inputs only.

Context: pipeline.py hardcodes BUY and decides intent by sentiment sign; Pipeline calls v2.1 per entity and does not connect PortfolioAllocator. Existing schemas include RLRawDecision/Action and PortfolioState. New whole-portfolio risk approval from T01 is mandatory.

Objective: Add a deterministic baseline decision adapter that consumes as-of market/context inputs and proposes a complete portfolio target map; pass through T01 risk approval. Keep RL policy optional and disabled by default.

Specification: Strategy protocol `propose_targets(as_of, market_state, portfolio_state, impact_records)->DecisionProposal`. Proposal contains strategy ID/version, inputs, target weights, reasons. MVP includes `NoTradeBaseline` (preserves current holdings) and `VolTargetBaseline`. For the latter, per instrument, calculate sample standard deviation of the trailing 20 daily close-to-close returns available before decision time, annualize by `sqrt(252)`, and set `target_weight = min(0.10, target_annual_vol / max(realized_annual_vol, 0.05))`; require at least 20 returns or keep current weight. `target_annual_vol` is explicit config, default 0.125. Do not use Impact Agent direction as a return forecast; unknown impact leaves baseline target unchanged. Decision maps unknown impact to no allocation change. Enforce long-only and no leverage. Pipeline constructor takes explicit config, policy and price provider; result includes proposal and RiskApproval objects. One portfolio request per cycle. No current hard-coded BUY.

Steps: inspect schemas and callsites; define typed proposal; implement no-trade and vol-target baselines; integrate with pipeline risk request; preserve old result fields as deprecated display adapters; add end-to-end offline tests.

Tests: no-trade holds; vol-target only uses past bars; negative/unknown impact does not trigger forced buy; multiasset proposal passed once to risk; risk rejection leaves no approved targets; deterministic repeated run; legacy pipeline smoke tests offline.

Acceptance: no unconditional BUY; every target originates from strategy and is approved portfolio-wide; no execution occurs in pipeline.

Exclusions: No PPO/SAC/CQL training, external data, order submission, return prediction, or hyperparameter search.

Budget: <=100 synthetic bars, <=10 assets, <30 sec/1GB.

Stop if existing interface consumers require unbounded behavior or target-weight semantics conflict with schemas; document adapter before breaking changes.

Report proposal contract, integration path, tests and legacy deprecations.
```

### T10 — Approval-bound mock order manager

```text
Task ID/version: T10 / v1
Title: Mock order manager with risk approval binding and reconciliation
Phase/priority: Phase 8, P1
Dependencies: T01 and T09 verified. Approval: local mock only; no broker/network.

Context: `ExecutionEngine.execute(orders)` accepts raw orders; PaperBroker fills immediately; AlpacaBroker is a dummy. Order and result schemas are section 4.6. No live broker integration is allowed.

Objective: Replace unsafe generic path with an order manager that derives mock order intents only from a current immutable RiskApproval and target delta, enforces idempotency, and reconciles mock positions/open orders.

Specification: Add `OrderManager.plan(approval, current_positions, prices, instrument_rules, as_of)` and `submit_mock(intents)`; reject if request hash/expiry/constraints version invalid, target differs from approval, price missing/stale, quantity non-finite/nonpositive, precision invalid, or intent ID reused with different payload. MockBroker supports accepted, rejected, partial fill, full fill, timeout->unknown, cancel, and query; deterministic scripted response. Unknown is reconciled before retry. Keep `AlpacaBroker` visibly disabled/stubbed and remove any implication `health=True` proves a connection. No raw public execute path may submit without approval; if compatibility retained, it must raise.

Steps: implement schemas; derive target deltas; add idempotency store local/in-memory; implement scripted mock lifecycle; reconciliation and kill switch; tests; update README labels.

Tests: missing/stale approval rejected; wrong hash and changed target rejected; duplicate same intent idempotent; same key/different payload rejected; partial/full/reject/timeout/reconcile; kill switch pauses new orders; positions reconcile; negative/short qty rejected.

Acceptance: no order can be submitted without matching live approval; mock lifecycle recoverable and deterministic; no network or real broker connection.

Exclusions: No Alpaca API calls, credentials, paper/live account, real order, market impact model, or async service.

Budget: <=10 synthetic orders, in-memory or pytest temp storage, <30 sec/1GB.

Stop if retaining the legacy execute path cannot be made fail-closed without broad breaking changes; stop and report callers.

Report APIs, lifecycle transitions, idempotency behavior, tests and proof broker transport was mocked.
```

### T11 — Point-in-time impact and strategy evaluation harness

```text
Task ID/version: T11 / v1
Title: Reproducible chronological impact-evaluation harness
Phase/priority: Phase 8, P2
Dependencies: T03, T04, T08, T09 verified. Approval: local synthetic labels only.

Context: Existing OOS artifacts are for v2 and not independently reproducible from checked-in raw data. No impact prediction evaluation harness exists. No performance claim is authorized.

Objective: Add a small offline evaluator for event-to-entity retrieval/classification and strategy target proposals with strict as-of filtering and reproducible metrics.

Specification: Input is versioned JSONL fixture with event/document availability timestamp, label creation timestamp, entity IDs and optional horizon outcome values. Exclude label from features until after its available time. Metrics: entity precision/recall/F1, Brier score only for calibrated probabilities (otherwise omit), coverage, abstention rate, target turnover, gross/net returns only if future outcomes are explicitly supplied in test fixture. Record dataset hash, code version, config and seed. Include naive/no-trade baseline. Report must label synthetic and non-investable. No parameter selection on test split.

Steps: define fixture schema; implement chronological split; implement evaluator; add leakage guard; emit JSON/Markdown summary to tmp_path in tests; document limitations.

Tests: label unavailable at decision time excluded; later-known relation excluded; repeated seed yields same output; baseline metrics; empty labels; metrics with no positives; train/test boundary; costs deducted if returns included.

Acceptance: evaluator never sees future labels/features; report reproducible from tiny fixtures; no result is described as validated alpha/profitability.

Exclusions: No real data, internet, model training, optimizer, large backtest, or tuning.

Budget: <=20 events, 10 entities, 3 horizons, <30 sec/1GB.

Stop if evaluation label semantics or horizon cannot be derived from supplied fixture; do not synthesize financial truth.

Report metrics, splits, dataset hash method, tests and explicit non-claims.
```

### T12 — Decision ledger and operational recovery

```text
Task ID/version: T12 / v1
Title: Append-only local decision ledger and health/recovery checks
Phase/priority: Phase 8, P2
Dependencies: T02, T09, T10 verified. Approval: pytest temporary SQLite only.

Context: Pipeline returns intermediate objects but does not durably record a complete decision→approval→order→fill chain. SQLite storage exists and T04 introduces versioned schema. Monitoring config is only a stub.

Objective: Persist enough local, redacted records to reproduce each mock decision and detect stale inputs, unapproved order attempts, incomplete orders and reconciliation mismatch.

Specification: Append-only records reference decision_time, source/document IDs, dataset/config/strategy versions, proposal hash, risk approval ID/hash/reasons, order intent/idempotency IDs, broker state transitions, fills/fees, portfolio snapshots, errors and recovery actions. Secrets are prohibited. Add health states `HEALTHY|DEGRADED|PAUSED`; stale/missing price, unknown broker order, failed storage write or risk config invalid → PAUSED. No automated trading side effect from alert. Provide query by cycle ID and deterministic replay serializer.

Steps: inspect T04 migrations; add ledger tables/repository; wire proposal/risk/mock lifecycle; implement health evaluation and redaction; tests using temp DB.

Tests: reconstruct complete mock cycle; append-only update behavior; secret-like field rejected/redacted; stale input pauses; unknown broker state pauses and requests reconciliation; ledger write failure fails closed; restart loads outstanding mock orders; no production DB path.

Acceptance: one mock cycle is reconstructable from persisted IDs/hashes; safety failures move health to PAUSED; no secret values stored.

Exclusions: No hosted observability, external messaging, production database, live broker, or automatic resume.

Budget: <=20 cycles, <=100 rows per test, temp DB only, <30 sec/1GB.

Stop if append-only guarantees conflict with current migrations or sensitive data appears in candidate payload; report and do not persist it.

Report schema, recovery state behavior, tests, and known retention gaps.
```

### T13 — Optional pretrained embedding model integration (approval required)

```text
Task ID/version: T13 / v1
Title: Integrate one pinned document embedding model for retrieval
Phase/priority: Phase 9, P2
Dependencies: T06 verified. Approval: NEEDS_APPROVAL; do not start until user explicitly approves this exact model download and bounded local inference operation.

Context: No embedding model or vector index exists. T06 provides provider interface and deterministic mock tests. Candidate models and limitations are section 4.8.

Before coding or downloading, request explicit approval including: exact Hugging Face repository/revision and license; expected download size; network requests; cache/model path; record count (initial maximum 5 synthetic documents); runtime/CPU/RAM; whether model execution is permitted. Approval to download does not authorize embedding real corpora.

Objective after approval: implement exactly one provider, selected by user approval, pinned to immutable revision, metadata complete per section 4.4. Do not train or fine-tune.

Specification: No silent remote download at runtime. Model must be explicitly preinstalled and offline load must fail closed if unavailable. Add model revision, tokenizer revision, dimension verification, max input tokens, preprocessing/chunker version, normalization and cosine metric. Store only synthetic fixture vectors initially. Separate query/document encoding instructions if model requires them. Run retrieval evaluation against labeled synthetic queries; compare to T06 mock/lexical baseline.

Tests: vector dimension/norm/finite checks; overlength chunk; no network when model absent; pinned model metadata; repeated deterministic embedding; small retrieval ranking; cache path; model revision mismatch rejected.

Acceptance: model is pinned and offline, retrieval results have source offsets and score semantics, evaluation is recorded; no economic prediction claim.

Exclusions: No bulk real document embedding, GPU, fine-tuning, vector database service, entity/event embeddings.

Budget: after approval, <=5 synthetic documents, <=20 chunks, one model download under separately approved size, CPU inference only unless separately approved, <2 minutes/2GB RAM.

Stop if license/revision/size differs from approval or model requires unapproved code/network. Completion report must state exact revision and operations.
```

### T14 — Optional real source connector pilot (approval required)

```text
Task ID/version: T14 / v1
Title: One-source bounded read-only data connector
Phase/priority: Optional Phase 9, P2
Dependencies: T05 verified. Approval: NEEDS_APPROVAL for exact source/request/records.

Before work, CEO presents source/provider, license/terms, endpoint, fields, requests, record cap, date window, data size, cost, timeout/retry, storage destination and deletion method. Wait for explicit approval for that operation. No broad scraping.

Objective after approval: integrate one named source as read-only adapter under existing adapter contract, with max_records and timeout; keep OFFLINE_MOCK default and connector disabled in config.

Specification: retain raw response hash, source/published/ingested/available timestamps, license metadata, provider IDs, validation and typed failure. Rate limit conservatively and make retries bounded. Never use fetched data for trading decisions in this task.

Tests: mock fixtures cover valid, empty, malformed, duplicate, 429/transient, timeout and source timestamp cases. Live smoke request only if approval explicitly includes that exact request and budget.

Acceptance: disabled by default; tests use mock transport; request/record limits enforced; one live operation only within written approval.

Exclusions: No second source, bulk history, scraping, paid endpoints without explicit cost approval, or broker action.

Budget: exact user-approved request and record ceiling; stop before exceeding it.

Stop if terms, rate limits, records, cost or endpoint differs from approved plan. Report exact actual count/cost/runtime.
```

### T15 — Optional supervised/RL research study (approval required)

```text
Task ID/version: T15 / v1
Title: Bounded offline model research against deterministic baselines
Phase/priority: Optional Phase 9, P3
Dependencies: T03, T09, T11 verified. Approval: NEEDS_APPROVAL for dataset, training, compute and runtime.

Before work, CEO requests explicit approval specifying dataset and license, data volume, train/validation/test dates, algorithm, CPU/GPU, estimated runtime, memory, output files, cost and no-promotion boundary. No RL because it exists in dependencies; justify simpler supervised/statistical alternatives first.

Objective after approval: run one reproducible offline research experiment only, compare with no-trade and deterministic baseline, use causal features and untouched final holdout, record code/data/config hashes and uncertainty.

Specification: No live or paper deployment. No model may alter default pipeline. No test-set tuning. Metrics include net returns only if execution costs and valid timestamps are supplied, plus drawdown, turnover, tail losses, calibration and ablations appropriate to task. Save artifact and evaluation report under approved local path.

Tests: split integrity, determinism, no future data, baseline parity, model reload, cost sensitivity. Tests themselves use tiny synthetic data.

Acceptance: complete reproducible report; model explicitly research-only; no promotion without separate review and approval.

Exclusions: No large sweep, online learning, automated promotion, external downloads, GPU-intensive training unless explicitly approved.

Budget: exact approved cap; default proposal if approved: CPU, <=10k rows, <=10 minutes, <=2GB RAM, zero network.

Stop if split leakage, data provenance uncertainty, resource cap approach, or unapproved package/model download occurs.

Report actual resource use, hashes, metrics, failed attempts and non-generalization caveats.
```

### T16 — Optional real broker integration (separate approval required)

```text
Task ID/version: T16 / v1
Title: Paper-account broker adapter behind approval-bound order manager
Phase/priority: Optional, P0 safety critical
Dependencies: T01, T09, T10, T12 verified. NEEDS_APPROVAL for any account connection. Live trading is not part of this task.

Before work, CEO must present broker, paper endpoint, account type, credential handling, exact read/write scopes, intended calls, request/rate cap, test orders (default zero), data/account exposure, logs, runtime, and disconnect/kill procedure. User approval must explicitly permit connection. Credentials must never be requested in chat or written to repo.

Objective after approval: implement a paper-only adapter with network transport mocked in ordinary tests, approval-bound order intents, idempotency, order lifecycle and reconciliation. Adapter must refuse live endpoint/config unconditionally.

Specification: credentials from approved secret manager/environment, redacted in logs; TLS; bounded timeouts; broker instrument precision; order status unknown on timeout then reconcile; no retry until client ID queried. Startup asserts paper URL/account and rejects live URL. Kill switch prevents submission. Persist broker IDs and fills through T12 ledger.

Tests: mock transport tests for accepted/rejected/partial/timeout/duplicate/cancel/reconcile; live endpoint rejected; missing secrets fails before network; kill switch; stale approval. If approval includes a zero-order paper health check, run only that check and record it; do not submit orders.

Acceptance: only paper endpoint is reachable under code; each submission is linked to current approval; no duplicate order on uncertainty; no live order path exists.

Exclusions: Live endpoint, real order submission, autonomous trading, credential sharing, account modification beyond separately approved paper test.

Budget: exactly user-approved requests, no orders unless separately explicit, bounded timeout and runtime.

Stop immediately on wrong endpoint/account, unapproved permission, unexpected order state, or credential/log exposure.

Report exact endpoints by environment name (not secrets), calls, order count, tests and reconciliation result.
```

### T17 — Benchmark-driven scale review

```text
Task ID/version: T17 / v1
Title: Measure MVP retrieval, graph and pipeline resource bottlenecks
Phase/priority: Phase 10, P3
Dependencies: T06, T07, T12 verified. Approval: local synthetic benchmark only.

Context: Repository has no scale measurements. Do not assume millions of entities, GPU, graph DB, or hosted vector search are needed.

Objective: Add a reproducible local benchmark and written decision whether current in-memory/SQLite design suffices at the measured MVP size.

Specification: Benchmark fixed synthetic sizes 10, 100, 1,000 entities/edges/chunks where feasible; record hardware/runtime version, wall time, peak RSS if available, exact query, build/run command, dataset seed. Measure SQLite insert/query, temporal graph traversal (<=3 hops), exact vector search at 384 dims only if T06 vectors exist, and pipeline cycle. Do not extrapolate to millions as fact; provide complexity-based caveats.

Steps: inspect current modules; create seeded generator; add standalone opt-in benchmark (not in unit-test suite); impose runtime ceiling; record results to stdout or user-approved report path; recommend next bottleneck only from measurements.

Tests: generator deterministic and bounded; benchmark smoke at 10 items in CI-safe unit test; no network; no persistent DB.

Acceptance: reproducible command, resource numbers and evidence-based recommendation; no infrastructure changes based solely on projected scale.

Exclusions: No large benchmark, cloud, hardware purchase, production migration, or million-node data generation.

Budget: default smoke <=1,000 records, <=60 sec, <=2GB; larger needs separate approval.

Stop if resource cap exceeded or benchmark would touch non-temporary storage.

Report setup, measurements, limits, and what evidence would justify scale work.
```

---

## 8. Hermes CEO operating instructions (copy-ready)

```text
You are the Hermes CEO/project manager for this trading-system project. Treat HERMES_TRADING_SYSTEM_EXECUTION_PACKAGE.md as the complete architecture and task source of truth. Read it fully at start and on every resume.

Operating loop:
1. Inspect current branch, git status, recent changes, task registry, and relevant tests before each delegation. Never assume state from a previous session.
2. Select exactly one task whose dependencies are VERIFIED and whose status is READY. Tasks marked NEEDS_APPROVAL cannot begin until the user approves that exact operation. Do not treat planning as approval.
3. Copy the complete prompt for that task from section 7 to one coder agent without combining tasks. Include current commit and any verified changes since package version 1.0. Do not paraphrase away scope, contracts, budgets or stop conditions.
4. Set registry state IN_PROGRESS and record coder assignment/date. Ask coder for the required completion report.
5. On return, set REVIEW. Independently inspect diff, relevant interfaces, side effects, and test output. Run only safe tests specified in prompt after verifying they do not access network, live accounts, or non-temporary DBs.
6. Check every acceptance criterion. Do not accept “tests pass” without inspecting test names/results and code. Record commit/hash, changed files, exact test command and outcome, reviewer evidence, and deviations in registry.
7. Mark VERIFIED only when evidence supports every criterion. Otherwise mark FAILED or BLOCKED and create one narrow corrective prompt referencing the failed criteria. Retry at most once without revising the plan; after a second failure, stop and update this package/architecture decision.
8. Delegate the next task only after current task is VERIFIED and all dependencies are VERIFIED. Preserve completed work; no broad rewrites.
9. If code changes invalidate a later prompt, stop delegation and update that prompt and dependency registry explicitly before continuing.
10. Stop on unexpected destructive side effects, production database access, external network use, live endpoint, unapproved cost, data-license uncertainty, secret exposure, or hard-risk bypass. Report what happened; do not conceal or automatically retry.

Safety/approval:
- Default OFFLINE_MOCK. The coder may use synthetic fixtures and pytest temp directories only.
- Before any approval-gated operation, present operation, purpose, source, request/record/data cap, estimated runtime/compute, cost, filesystem/database effects, license/safety considerations, and exact permission requested. Wait for explicit approval for that operation. One approval never authorizes later operations.
- Never authorize live trading. This CEO package has no live trading milestone. T16 is paper-only and still requires a separate explicit approval; it may not add a live endpoint.
- Never request or display secret values. Do not contact third parties or external services without approval.

Task tracking:
- Update the registry in this same persistent document after each task: status, prompt version, branch/commit, files actually changed, tests and exact results, approval reference if relevant, acceptance evidence, deviations, and remaining issue.
- If the document cannot be edited, keep work stopped until a durable registry is available; do not claim cross-session persistence.
- Keep one coder task active at a time. Do not send a future task prompt early.

Project completion requires all required offline MVP tasks T01–T12 VERIFIED, no open P0/P1 blockers, clean reproducible tests, no default network/trading path, and an evidence-backed final summary. T13–T17 are optional and do not block MVP completion unless scope is explicitly expanded.
```

---

## 9. Testing and verification protocol

1. Inspect each test’s side effects before running it. Exclude network-dependent tests unless an exact external operation has been approved.
2. Use `./.venv/bin/pytest <focused-tests>` in this environment; do not use `uv run` if its cache remains read-only. Use `tmp_path` for all DB tests.
3. Test contracts in layers: schema validation, module logic, integration with offline fakes, and deterministic replay. A source file or class is not proof of integration.
4. New tests must cover valid, boundary, invalid, stale/missing input, and failure/recovery cases as appropriate. Include negative tests proving default mode does not call network or broker transports.
5. For time-series/model evaluation, test future perturbation, timestamp cutoffs, train/validation/test separation, and reproducibility. Preserve the final holdout from tuning.
6. Report test count, pass/fail/skip, exact command, warnings, network attempts, and generated files. Never invent a test result.
7. Do not run training, full OOS experiments, web UI server, scraping, bulk ingestion, or live broker test as a routine verification step.

---

## 10. Safety, resources and approval policy

### Default allowed

- Offline mocks, synthetic data, bounded unit tests, and isolated temporary databases after inspecting the command.
- Tiny fixtures up to 10 synthetic entities, 10 graph edges, 5 documents, 20 mock chunks/vectors and 100 market bars by default. Use less when enough.
- CPU-only, deterministic, under 30 seconds and 1 GB RAM for ordinary task tests unless prompt has a specific approved bound.

### Requires exact user approval

- Any network access, real RSS/market data retrieval, bulk historical downloads, broad scraping, paid APIs, pretrained model download or real-corpus embedding.
- Long-running computation, GPU use, RL/model training, hyperparameter/Monte Carlo sweeps beyond bounded unit fixtures.
- Any DB outside pytest temp directory, migration of user/production data, external account connection, paper account order, live broker connection or order.
- Hardware/cloud/infrastructure purchases or hosted services.

The approval request must give operation, purpose, sources, exact requests/records/data size, runtime/compute, cost, storage effects, license/security considerations and clear permission requested. Stop until explicit user approval. If a command unexpectedly attempts a network request or touches an unintended DB, terminate safely and report.

Do not claim network or trading is technically disabled just because config says so. Verify the code boundary. Current repository has no real Alpaca connection, but generic execution is callable without risk approval; future code must enforce the boundary.

---

## 11. Failure recovery and corrective tasks

- **Test failure:** classify as code failure, environment limitation, or unapproved external dependency. Preserve output. Do not edit unrelated code to make a suite green. Create a corrective task for the narrow failing criterion.
- **Task exceeds scope:** stop, keep current diff reviewable, and ask CEO to split/resequence; do not fold the next roadmap task into it.
- **Architecture conflict:** stop before changing public contracts. CEO updates this package with the decision, affected prompts and registry dependencies.
- **Unexpected data/DB/network side effect:** stop the process, do not retry, record exact command/target/observed effect, assess whether rollback is safe. Never delete data to hide a failure.
- **Coder fails acceptance:** CEO marks FAILED, includes concrete evidence, issues one corrective prompt limited to the acceptance gap. If corrective attempt fails, mark BLOCKED and revise plan or request user decision.
- **New defect discovered:** severity and affected paths go in registry; if P0/P1, pause dependent tasks until risk is resolved.
- **Partial task completion:** do not mark VERIFIED. Keep a clear list of completed subcriteria and remaining work; the retry prompt must avoid rewriting verified portions.

---

## 12. Resume and progress tracking

This Markdown file is the persistent task registry. On each session restart, CEO must:

1. Read registry and package version.
2. Check `git status`, HEAD, and whether this file has later updates.
3. Reconcile task status against actual code/tests; downgrade unsupported VERIFIED states to REVIEW.
4. Confirm dependency completion and approval status.
5. Delegate only the single next eligible task.

Registry evidence update format:

```text
Task ID / prompt version:
Status:
Branch / commit:
Files changed:
Tests (exact command and counts):
Acceptance evidence:
Approval reference and bounded operation (if applicable):
Deviations / blockers:
Reviewer and date:
```

Append task completion notes below the table in section 6; update that table status. Do not claim the plan is persistently tracked in any other system unless such system is actually connected and the record is verified.

---

## 13. Future activation gates

- **External data:** T14 only after explicit source-specific approval; adapter stays disabled by default. Validate license, timestamp, coverage, revisions, missingness and point-in-time availability before modeling.
- **Pretrained embeddings:** T13 only after model/revision/license/download approval. A small retrieval benchmark must beat a lexical baseline on labeled queries before adoption. No real-world bulk embedding without a separate approval.
- **Supervised/RL training:** T15 only after causal simulator and evaluation gates pass. Research artifact cannot be loaded into production pipeline without an independent promotion review and separate explicit approval.
- **Paper broker:** T16 is optional, requires previous MVP safety gates and exact account/scope approval. Start with read-only health check if possible; order count defaults to zero. A paper approval is not live authorization.
- **Live trading:** Not included in this package. Requires a separately designed safety review, user instruction, and explicit approval. No task here may enable it.
- **Scaling:** T17 must measure current bottlenecks first. New databases, GPU, hosted vector service or cloud resources require demonstrated benchmark need and explicit approval if cost/data exposure applies.

---

## 14. Definition of MVP completion

Required offline MVP is complete only when T01–T12 are VERIFIED with recorded evidence, and:

- All risk checks operate on a whole proposed portfolio and enforce explicit hard limits.
- Pipeline loads validated config, uses point-in-time inputs, updates risk history, and does not hard-code BUY.
- Every approved target is traceable to a deterministic strategy proposal and risk approval.
- RL/backtest timing tests prove no future leakage; reported backtest outputs are net of explicit configured costs or clearly gross.
- Catalog, source provenance, event evidence and temporal graph relations are versioned and reproducible on small local fixtures.
- Impact Agent outputs are evidence-linked; retrieval similarity and sentiment are not presented as causal or predictive evidence.
- Mock orders cannot bypass approval; duplicate/unknown order states are recoverable and auditable.
- Decision records can reconstruct a mock cycle; health pauses on stale/missing critical inputs or reconciliation uncertainty.
- Default operation is offline mock and has no real broker or live trading route.
- Safe offline tests pass; external-data test failures are separated and reported, not hidden.
- There are no open P0/P1 defects, and the final CEO review documents limitations and unresolved optional approvals.

This definition does not imply profitability, production readiness for a real broker, or validated causal impact estimates.

---

## 15. First delegation

The CEO should delegate **T01 — Portfolio-level v2.1 risk contract and hard limits** first. It addresses the most important verified safety gap, defines the whole-portfolio approval object needed by pipeline and order work, and can be completed and tested entirely offline with synthetic inputs. T03 may proceed independently afterward or in another task slot only after the CEO has verified T01; this package still directs one active coder task at a time.

**Unresolved assumptions:** instrument metadata and sector membership must be supplied by the future catalog rather than guessed; market-data freshness thresholds need explicit source-specific configuration; any relation strength or impact confidence remains uncalibrated until labeled historical evaluation exists; model download/license details must be rechecked at T13; OOS artifacts for v2.1 are not present as reproducible raw-data results in the checked-in `var/forensic_v21` directory.

---

## 16. CEO architecture decisions (append-only)

### AD-01 — T02 config validation vs. partial-config callers (2026-10-13)

**Context:** T02 requires both "validate configuration at pipeline construction and fail closed on a
missing required risk limit" AND "preserve `Pipeline(config=...)` callers". Two coder attempts
revealed the conflict: strict construction-time validation fails closed for existing callers that
pass a partial config dict (e.g. `Pipeline({"closes_provider": stub})` in
`tests/test_multi_entity.py`), which broke 2 previously-passing tests.

**Decision:** `Pipeline.__init__` builds an **effective config by deep-merging the repository config
(`load_config()`) as the base with the caller's explicit config overlaid** (recursive dict merge;
non-dict values such as a callable `closes_provider` from the caller win). Required risk limits are
then validated on the *effective* config and the pipeline **fails closed** if a required limit is
still missing/non-finite after the merge. `Pipeline(config=None)` → effective config = repo config.
This preserves existing callers (they inherit validated repo risk defaults) while keeping the
fail-closed guarantee for a genuinely incomplete effective config.

**Affected prompts/registry:** T02 (T02-c1 and the final T02-c2 corrective); T02 status BLOCKED
pending the corrective. T01 unaffected. Follow-on tasks (T03+) unaffected; the T02 dependency
contract (validated risk settings reachable via `pipeline.risk.cfg`) is unchanged.

