# Concept — Catalog + RAG for the Impact Agent

**Status: concept / design only. Nothing here is implemented yet. No code, no schema
migration, no cron.** This document describes *how it should work* so it can be reviewed
before anything is built.

The goal is to make the Impact Agent stop *guessing* which instrument an event hits, and
instead **resolve** it against a maintained, deterministic catalog of instruments,
entities and relations — with the language model only assigning sentiment/confidence on
a grounded candidate set.

---

## 1. Why — the two problems this fixes

1. **World knowledge.** A model (especially a distilled small one) knows the *form* of the
   task but not the *facts*: it does not reliably know that Jensen Huang → NVDA, that the
   EU → ASML/STM/SAP (European names first), or that Mistral is **not listed** (so there is
   no ticker). On rare/new events it invents a ticker. The damage in a trading bot is in the
   **tail** (one wrong ticker), not in average latency.
2. **Non-reproducible RL input.** If the model *generates* the per-asset vector, the same
   event yields a different vector tomorrow. The RL model cannot train on that, and backtests
   cannot be compared (leakage + noise). The vector must be deterministic.

Both are solved by moving facts into a **database** and using the model only for
interpretation (sentiment/confidence) over a **pre-resolved candidate set**.

---

## 2. Core principle

> **impact = theme-match × region-match × relation-distance**

"EU announces a €50bn AI investment" must hit *Mistral* (EU + AI, direct) **and**
*NVDA/AMD* (AI, indirect, non-EU). That is only possible if the DB knows *what* an
instrument is exposed to, *where* it sits, and *how* companies/people connect.

This is a **graph + taxonomy** model, not a flat "ticker → sector" table.

---

## 3. Keep three things apart (this is the key correction)

| | What it is | Where it belongs |
|---|---|---|
| **The 60-vector / categorization** | identity + profile of an instrument | **deterministic catalog (DB)** |
| **Currency (up-to-date facts)** | which facts/links are true now | **RAG / context injection (DB → prompt)** |
| **Behavior** | JSON contract, using the injected context | **(optional) LoRA fine-tune** |

Consequences:

- The **vector comes from the DB, never from the model**. Reproducible, versionable,
  testable — exactly what the RL model and the backtester need.
- **Training does not make a model current.** Currency comes from injecting DB rows into
  the prompt (RAG), not from weights. A LoRA can only teach *behavior* (follow the output
  contract, consume the injected context) — it is optional and comes last.
- **Time-version everything** (`as_of`, `valid_from/valid_to`). Sector/region change rarely,
  but size/market-cap and relations change often; injecting *today's* market cap into a
  2018 replay leaks the future into the RL model.

---

## 4. Database schema

SQLite (consistent with the rest of the project). All tables carry an `as_of` /
`valid_from`–`valid_to` where the value can change over time.

### 4.1 Taxonomy — the "systematic naming" (controlled vocabularies)

- **`sectors`** — GICS-like hierarchy: *Information Technology → Semiconductors &
  Semiconductor Equipment → …*. Hierarchical so you can generalize (a "semiconductor" is a
  kind of "tech") and be specific.
- **`regions`** — country (ISO) → region (Europe, North America, APAC) → **bloc**
  (EU, US, China, ASEAN). Events talk at bloc level, so this layer is required.
- **`asset_classes`** — stock, bond, ETF, commodity, fx, crypto.
- **`themes`** — *cross-cutting*, the key to indirect impact: `AI`, `semiconductors`,
  `datacenters`, `energy`, `defense`, `EV`, `quantum`, … A theme runs across sectors
  (AI touches software, chips, power, cooling).

### 4.2 Instruments + their profile vector

- **`instruments`** — `ticker, isin, name, exchange, country, asset_class, currency,
  is_listed, is_active, as_of`.
- **`instrument_sectors`** — many-to-many (a conglomerate spans sectors) with a
  weight / primary flag.
- **`instrument_exposures`** — ⭐ the indirect-impact engine: per instrument per **theme**
  a weight + `region_scope`. E.g. `NVDA → AI (1.0, global)`,
  `ASML → semiconductors (1.0, global) + EU presence`.
- **`profiles`** — the derived **60-vector** (asset-class one-hot, sector one-hot,
  region one-hot, size bucket, vol bucket, theme-exposure vector). Generated *from the
  tables above*, not by the model, with `as_of` for backtest reproducibility.

### 4.3 Entities + relations (the graph — this is where "Jensen Huang = CEO of NVDA" lives)

An **entity ≠ an instrument**. Mistral is an entity *without a ticker* (not listed) — it
must be in the DB, otherwise the model hallucinates a ticker for it (as `MLST` happened
before).

- **`entities`** — `id, name, type (person|company|government|institution|region),
  country, instrument_id (nullable — set when listed)`.
- **`persons`** — role data (Jensen Huang → role CEO, at NVDA).
- **`entity_relations`** — ⭐ the graph:
  `subject, predicate, object, valid_from, valid_to, strength, source, confidence`
  with predicates: `CEO_OF`, `WORKS_AT`, `FOUNDED`, `OWNS`, `SUBSIDIARY_OF`,
  `SUPPLIER_TO`, `COMPETITOR_OF`, `REGULATES`, `MEMBER_OF`, `LOCATED_IN`.
  Rows: `Jensen Huang — CEO_OF → NVDA`, `ASML — SUPPLIER_TO → TSMC`,
  `TSMC — SUPPLIER_TO → NVDA`, `Mistral — LOCATED_IN → FR`.
  This gives **second-order propagation**: EU AI plan → demand for AI chips →
  supply chain → NVDA/AMD/TSM/ASMI.

### 4.4 Current / fundamental data (the "up-to-date" part)

- **`fundamentals`** — `market_cap, price, vol_30d, last_earnings, as_of`.
- **`revenue_geo`** — ⭐ revenue by region. **This is the field that makes it really
  work:** a policy hits a company more the more revenue comes from that region. NVDA's
  EU revenue share decides whether it gets +0.4 or +0.05.
- **`revenue_segment`** — revenue by segment/theme (how much of NVDA is actually AI).
- **`macro_indicators`** — for macro events (rates, inflation).

### 4.5 Aliases (entity resolution)

- **`aliases`** — `entity_id, alias, lang, type (person|ticker|brand|abbreviation)`.
  `Jensen Huang` · `Huang` · `NVIDIA` · `NVDA` · `the green team` → all NVDA. This is what
  deterministically ties "Jensen Huang" to a ticker — **no model knowledge**.

### 4.6 Provenance & audit

- **`sources`**, **`updates_audit`** (`old → new, source, ts`), and
  **`impact_examples`** (a gold few-shot / hold-out set for the model and for evaluation).

---

## 5. How the DB answers the example

**"The EU announces a €50bn AI investment"**

1. **Parse** (code): theme = `AI`/`semiconductors`, region = `EU`, actor = government.
2. **Candidate set** = union of:
   - instruments with a high **theme exposure** to `AI` → NVDA, AMD, ASML, STM, SAP…
   - everything in the **EU region** within that theme → ASML, STM, SAP
     (Mistral = entity without ticker: named, not tradeable).
3. **Rank by distance:**

   | distance | meaning | example |
   |---|---|---|
   | 0 | the event names the instrument | — |
   | 1 | theme + region match | ASML, STM, SAP (EU + AI) |
   | 2 | via **relation** (supplier/customer/subsidiary) | TSM, ASMI |
   | 3 | theme only, non-EU | NVDA, AMD (+0.4) |

4. **Inject** the top-N compactly (name, sector, region, exposure weight, EU revenue share)
   into the prompt → the **LLM only assigns sentiment/confidence**. It no longer guesses
   a ticker.
5. **Vector from `profiles`** → the RL model. Reproducible, no model-invented numbers.

### The pipeline, end to end

```
event text
  → entity extraction (gazetteer: aliases + optional NER)     ← deterministic, code
  → candidate tickers from the catalog
  → inject compact context: "CANDIDATES: NVDA (semiconductor, US, mega-cap, AI 1.0) …"
  → LLM: pick + sentiment/confidence                          ← the ONLY model step
  → validate tickers against the catalog whitelist
  → vector from profiles → RL
```

This is the "90% → closer to 100%" jump: the model **chooses from a grounded candidate
set** instead of guessing. Be honest though: it is not literally 100% — the residual risk
moves to *catalog quality* (a missing alias, a stale row). So measure on a hold-out, and
keep the keyword fallback + whitelist validator.

---

## 6. Design rules (do not break these)

- **Relations and revenue-geo are time-versioned.** A CEO changes, a supplier changes,
  revenue shares shift — otherwise the future leaks into the backtest.
- **The LLM never writes the vector or the relations.** It only does
  `event → entities + sentiment`.
- **The curator never writes the DB unchecked.** A stronger model *proposes* updates with a
  cited source; a deterministic validator (does the ticker exist? is the sector valid? is the
  alias unique?) approves; only then is the row written, with an audit entry. Otherwise you
  persist hallucinations.
- **Keep the keyword fallback and the whitelist validator** as the backstop.

---

## 7. Minimal viable vs. growth

**V1 (delivers ~90% of the value):** `instruments`, `sectors`, `regions`, `themes`,
`instrument_exposures`, `entities`, `aliases`, `entity_relations`
(only `CEO_OF` / `SUPPLIER_TO` / `LOCATED_IN`). This already derives
"EU AI → Mistral + ASML + NVDA".

**Growth later:** `revenue_geo` / `revenue_segment` (makes indirect weighting sharp),
full supply chain, ownership, macro.

**Most of the upkeep is in** revenue-geo, relations and aliases — exactly the three the
curator loop (weekly: diff → propose → validate → audit) keeps current. Sectors / regions /
themes are stable and rarely change.

---

## 8. What this document is NOT

No code, no schema migration, no cron, no seed loader, no changes to `impact.py`.
This is the design to review first. Implementation, in the recommended order, would be:

1. Schema + seed loader (build the catalog, compute the vectors).
2. Resolver + context injection + whitelist validation in the impact path.
3. Curator loop (weekly, silent).
4. *(optional, last)* LoRA fine-tune — only if the hold-out numbers justify it.

Train nothing before step 1–2 are measured: a pretrained local model (e.g. Qwen3-8B) with
the injected catalog context needs no training to be current — the DB provides currency,
not the weights.
