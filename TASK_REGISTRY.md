# Hermes Trading System — Task Registry (durable)

**Source of truth for the plan:** `HERMES_TRADING_SYSTEM_EXECUTION_PACKAGE.md` (package v1.0, prepared 2026-10-13).
**This file** is the durable, cross-session task registry required by the CEO operating instructions (package §8, §12).
**Baseline repository:** branch `master`, HEAD `70be28d` ("Concept: Impact Agent catalog + RAG design").

**Delegation / coder route (BINDING):** all coding tasks are executed by the specialised
**`coder` consultant** (Atlas Hermes profile `coder`, model `qwen/qwen3-coder-30b-a3b-instruct`,
`toolsets: [kanban]` + terminal), dispatched over the shared kanban board — NOT via generic
`delegate_task` subagents. Standard dispatch:

```bash
hermes kanban create "<title>" --assignee coder \
  --workspace dir:/home/olivier/trading-bot \
  --body-file <task_body.md> --created-by ceo --idempotency-key <key>
hermes kanban dispatch     # spawns 'hermes -p coder ... kanban task <id>'
```

After dispatch, verify the worker really runs (`ps aux | grep "hermes -p coder"`) and that new
files/commits appear in the repo — a "running" status without a process is a zombie claim.
T01 was implemented before this route was fixed and by a generic subagent; it was independently
verified by the CEO (39 tests + 10/10 acceptance checks, commit `4f27220`) and stands on that
evidence. From T02 onward, coding runs only through the `coder` consultant.

Allowed states: `BACKLOG`, `READY`, `IN_PROGRESS`, `REVIEW`, `VERIFIED`, `BLOCKED`, `NEEDS_APPROVAL`, `FAILED`.

## Rules (unchanged from package)

- One active coder task at a time. Only tasks whose dependencies are `VERIFIED` and whose own state is `READY` may be released.
- `NEEDS_APPROVAL` tasks cannot start until the user explicitly approves that exact, bounded operation. Planning is not approval.
- A task is `VERIFIED` only when the reviewer has independently inspected the diff, interfaces, side effects and test output and every acceptance criterion is met.
- Default operation is `OFFLINE_MOCK`: synthetic fixtures and pytest `tmp_path` only. No network, no real data, no model training, no broker, never live trading.

## Status table

| ID | Task | Priority | Dependencies | Status | Prompt ver. | Branch/commit | Files changed | Tests (exact) | Approval | Evidence / notes |
|---|---|---:|---|---|---|---|---|---|---|---|
| T01 | Portfolio-level v2.1 risk contract and hard limits | P0 | None | VERIFIED | v1 | master @ 4f27220 (committed) | schemas.py, risk_v2_1/portfolio_risk.py (new), risk_v2_1/__init__.py, tests/test_risk_v2_1.py | `pytest tests/test_risk_v2_1.py tests/test_core.py` = 39 passed | none needed (offline) | CEO-verified 2026-10-13; 10/10 independent acceptance checks pass |
| T02 | Validated config and pipeline risk-state feed | P0 | T01 | VERIFIED | v1 | master @ c25046d (committed) | config/config.yaml, hermes_bot/pipeline.py, hermes_bot/risk/__init__.py, tests/test_pipeline.py | mandated `pytest tests/test_pipeline.py tests/test_risk_v2_1.py` = 45 passed; broad offline green; ruff clean | none | CEO-verified 2026-10-13 after AD-01 (deep-merge); probe A–E pass; attempts t_b44267f0→t_46972957→t_153095e7→t_b36f6754 |
| T03 | Causal RL environment and cost-aware backtest timing | P0 | None | VERIFIED | v1 | master @ 88ce8ea (committed) | backtest/__init__.py, backtest/metrics.py, backtest/runner.py, rl/env.py, train_rl.py, tests/test_backtest.py, tests/test_rl_env.py | mandated `pytest tests/test_backtest.py tests/test_rl_env.py tests/test_mc_wiring.py` = 31 passed; broad offline green; ruff clean | none | CEO-verified 2026-10-13; causal invariants + cost reconciliation probed; t_83fe72ae (blocked) → t_94d1c88d (c1 GREEN), both on kimi-k2.7-code |
| T04 | Canonical catalog, provenance models and SQLite migrations | P1 | T01 contracts | VERIFIED | v1 | master @ 720616d (committed) | data/storage.py, data/catalog.py (new), data/migrations.py (new), schemas.py, tests/test_data.py | `pytest tests/test_data.py` = 10 passed; broad offline green; ruff clean | none | CEO-verified 2026-10-13; fresh[0,1]+idempotent+legacy-preserved+rollback+FK probed; var DB untouched; kanban t_62ea4571 |
| T05 | Offline-first ingestion interfaces and timestamp normalization | P1 | T04 | IN_PROGRESS | v1 | master @ 720616d | — | — | none | dispatched to coder 2026-10-13 (T04 VERIFIED) |
| T06 | Document chunking/retrieval interface and mock vector index | P2 | T04, T05 | BACKLOG | v1 | — | — | — | none | Blocked on T04,T05 |
| T07 | Typed temporal economic graph store and bounded traversal | P2 | T04 | BACKLOG | v1 | — | — | — | none | Blocked on T04 |
| T08 | Deterministic evidence-linked event/impact extraction | P1 | T04, T05, T07 | BACKLOG | v1 | — | — | — | none | Blocked on T04,T05,T07 |
| T09 | Decision engine and deterministic baseline target allocator | P1 | T01, T02, T03, T08 | BACKLOG | v1 | — | — | — | none | Blocked on T01,T02,T03,T08 |
| T10 | Approval-bound mock order manager and reconciliation | P1 | T01, T09 | BACKLOG | v1 | — | — | — | none | Blocked on T01,T09 |
| T11 | Point-in-time impact and strategy evaluation harness | P2 | T03, T04, T08, T09 | BACKLOG | v1 | — | — | — | none | Blocked on dependencies |
| T12 | Decision ledger, health metrics and recovery protocol | P2 | T02, T09, T10 | BACKLOG | v1 | — | — | — | none | Blocked on T02,T09,T10 |
| T13 | Optional pretrained embedding model integration | P2 | T06; user approval | NEEDS_APPROVAL | v1 | — | — | — | REQUIRED (model download) | Do not start without exact approval |
| T14 | Optional real source connector pilot | P2 | T05; user approval | NEEDS_APPROVAL | v1 | — | — | — | REQUIRED (source/requests) | Do not start without exact approval |
| T15 | Optional supervised or RL research study | P3 | T03, T09, T11; user approval | NEEDS_APPROVAL | v1 | — | — | — | REQUIRED (data/training/compute) | Do not start without exact approval |
| T16 | Optional real broker integration (paper only) | P0 safety | T01, T09, T10, T12; separate approval | NEEDS_APPROVAL | v1 | — | — | — | REQUIRED (account connection) | Paper only; live never included |
| T17 | Benchmark-driven scale review | P3 | T06, T07, T12; measurements | BACKLOG | v1 | — | — | — | none | Blocked on dependencies |

## Task completion notes (append below, newest first)

### T03 — Causal RL environment and cost-aware backtest timing

```text
Task ID / prompt version: T03 / v1  (attempts: t_83fe72ae BLOCKED, t_94d1c88d GREEN)
Status: VERIFIED (commit 88ce8ea)
Branch / commit: master @ 88ce8ea (committed)
Files changed: hermes_bot/backtest/__init__.py, backtest/metrics.py, backtest/runner.py,
  hermes_bot/rl/env.py, hermes_bot/train_rl.py, tests/test_backtest.py, tests/test_rl_env.py
Tests: mandated `pytest tests/test_backtest.py tests/test_rl_env.py tests/test_mc_wiring.py`
  = 31 passed. Broad offline suite = all passed (exit 0). ruff clean.
Acceptance evidence (CEO probes): momentum trailing-only (perturbing prices after t leaves
  mom[:t+1] unchanged); env observation at t identical under future perturbation; next-bar reward
  hand-check (action=1 -> alloc 0.5, first return 0.05 = 0.5*(110-100)/100); cost reconciliation
  (net_returns == equity path exactly; sum(gross)-sum(net)=0.00150226 ~ total_costs/initial=0.0015023).
Approval reference and bounded operation: none (offline synthetic only)
Deviations / blockers: 1st attempt blocked (capability) on default coder model qwen3-coder-30b-a3b
  with no changes -> re-dispatched with task-level model override moonshotai/kimi-k2.7-code.
  2nd produced correct causal fixes but a net-cost accounting bug (cumulative total_costs subtracted
  per bar -> sum(gross-net) 7x too large); fixed in corrective T03-c1. Coder left no scratch files.
Reviewer and date: CEO agent, 2026-10-13
```

### T02 — Validated config and pipeline risk-state feed

```text
Task ID / prompt version: T02 / v1  (attempts: t_b44267f0 FAILED, t_46972957 FAILED, t_153095e7 AD-01, t_b36f6754 GREEN)
Status: VERIFIED (commit c25046d)
Branch / commit: master @ c25046d (committed)
Files changed: config/config.yaml, hermes_bot/pipeline.py, hermes_bot/risk/__init__.py, tests/test_pipeline.py
Tests: mandated `pytest tests/test_pipeline.py tests/test_risk_v2_1.py` = 45 passed.
  Broader offline (core/pipeline/risk_v2/risk_v2_1/multi_entity/backtest/mc_wiring/sentiment/
  indicators/montecarlo/impact/b2b/trained_policy) = all passed (exit 0). ruff clean.
Acceptance evidence (final CEO probe): (A) Pipeline() risk.cfg["max_asset_weight"]==0.1;
  (B) Pipeline({"closes_provider": stub}) constructs OK (partial callers preserved);
  (C/D) hist_returns grows once per mark (0->2); (E) genuinely incomplete EFFECTIVE config ->
  PortfolioMarkError (fail closed). Canonical §4.7 limits present in config/config.yaml.
  No network/broker refs in changed files; duplicated/historically failed attempts recorded.
Approval reference and bounded operation: none (offline synthetic only)
Deviations / blockers: resolved via AD-01 (deep-merge caller config over repo defaults before
  validation). One coder-authored test encoded pre-AD-01 semantics and was realigned in the final
  corrective. Coder consultant left debug scratch files in the repo root on each run (CEO cleaned).
Reviewer and date: CEO agent, 2026-10-13
```

### T01 — Portfolio-level v2.1 risk contract and hard limits

```text
Task ID / prompt version: T01 / v1
Status: VERIFIED
Branch / commit: master @ 70be28d (changes uncommitted)
Files changed: hermes_bot/schemas.py (+RiskStatus/RiskTargetPosition/RiskRequest/RiskCheck/RiskApprovalResult);
  hermes_bot/risk_v2_1/portfolio_risk.py (new evaluator); hermes_bot/risk_v2_1/__init__.py
  (approve_portfolio + clock + approval_ttl_seconds; approve() marked LEGACY, body unchanged);
  tests/test_risk_v2_1.py (+15 portfolio tests)
Tests (exact command and counts): `./.venv/bin/pytest tests/test_risk_v2_1.py tests/test_core.py` -> 39 passed in 0.20s;
  broader offline `pytest tests/test_risk_v2.py tests/test_pipeline.py tests/test_multi_entity.py
  tests/test_backtest.py tests/test_mc_wiring.py` -> 27 passed in 1.20s. No network/broker refs in changed files.
Acceptance evidence: CEO independent check (/home/olivier/.hermes/cache/scratch/t01_verify.py) 10/10 PASS:
  valid-under-limits approved; single-name cap reduced (peak 0.10); aggregate gross <= cap; short request rejected;
  9>8 positions rejected; NaN rejected; missing hard config -> paused with explicit reason codes;
  permuted input -> identical request_hash + targets (order-independent); result immutable; stale price -> paused.
  All acceptance criteria in prompt met.
Approval reference and bounded operation (if applicable): none (offline synthetic tests only)
Deviations / blockers: non-breaking, documented: (1) added optional `universe_ids` to RiskRequest (contract gives no ID
  resolver); (2) `paused` for missing-config/stale-data, `rejected` for invalid/unsupported input; (3) sector/position-count
  overflow -> reject (no scaling rule in spec); (4) added `approval_ttl_seconds` + injectable `clock` (clock explicitly
  required by spec). No stop conditions; additive schema only. config/config.yaml not extended (that is T02).
Reviewer and date: CEO agent, 2026-10-13
```
