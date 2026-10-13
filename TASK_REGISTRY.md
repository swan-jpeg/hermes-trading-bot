# Hermes Trading System — Task Registry (durable)

**Source of truth for the plan:** `HERMES_TRADING_SYSTEM_EXECUTION_PACKAGE.md` (package v1.0, prepared 2026-10-13).
**This file** is the durable, cross-session task registry required by the CEO operating instructions (package §8, §12).
**Baseline repository:** branch `master`, HEAD `70be28d` ("Concept: Impact Agent catalog + RAG design").

**Delegation / coder model:** the coder subagent is pinned in `~/.hermes/config.yaml` to
`delegation.provider=nous`, `delegation.model=qwen/qwen3-coder-30b-a3b-instruct` (coding specialist).
Before this pin, subagents inherited the parent model (`deepseek/deepseek-v4.1-flash`). T01 was
written by an inherited-model child but was independently verified by the CEO (39 tests + 10/10
acceptance checks) and stands on that evidence. Verify the child model with `delegate_task(action='list')`
at each spawn.

Allowed states: `BACKLOG`, `READY`, `IN_PROGRESS`, `REVIEW`, `VERIFIED`, `BLOCKED`, `NEEDS_APPROVAL`, `FAILED`.

## Rules (unchanged from package)

- One active coder task at a time. Only tasks whose dependencies are `VERIFIED` and whose own state is `READY` may be released.
- `NEEDS_APPROVAL` tasks cannot start until the user explicitly approves that exact, bounded operation. Planning is not approval.
- A task is `VERIFIED` only when the reviewer has independently inspected the diff, interfaces, side effects and test output and every acceptance criterion is met.
- Default operation is `OFFLINE_MOCK`: synthetic fixtures and pytest `tmp_path` only. No network, no real data, no model training, no broker, never live trading.

## Status table

| ID | Task | Priority | Dependencies | Status | Prompt ver. | Branch/commit | Files changed | Tests (exact) | Approval | Evidence / notes |
|---|---|---:|---|---|---|---|---|---|---|---|
| T01 | Portfolio-level v2.1 risk contract and hard limits | P0 | None | VERIFIED | v1 | master @ 70be28d (uncommitted) | schemas.py, risk_v2_1/portfolio_risk.py (new), risk_v2_1/__init__.py, tests/test_risk_v2_1.py | `pytest tests/test_risk_v2_1.py tests/test_core.py` = 39 passed | none needed (offline) | CEO-verified 2026-10-13; 10/10 independent acceptance checks pass |
| T02 | Validated config and pipeline risk-state feed | P0 | T01 | IN_PROGRESS | v1 | master @ 70be28d | — | — | none | Delegated to coder-agent 2026-10-13 (T01 VERIFIED) |
| T03 | Causal RL environment and cost-aware backtest timing | P0 | None | READY | v1 | — | — | — | none needed (offline) | Independent; held until T01 verified (one task at a time) |
| T04 | Canonical catalog, provenance models and SQLite migrations | P1 | T01 contracts | BACKLOG | v1 | — | — | — | none | Blocked on T01 |
| T05 | Offline-first ingestion interfaces and timestamp normalization | P1 | T04 | BACKLOG | v1 | — | — | — | none | Blocked on T04 |
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
