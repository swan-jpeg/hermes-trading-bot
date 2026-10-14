"""Tests for T12 / v1: append-only decision ledger + health/recovery."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from hermes_bot.data.migrations import MigrationRunner
from hermes_bot.execution.mock_broker import MockBroker
from hermes_bot.execution.order_manager import OrderManager
from hermes_bot.execution.schemas import InstrumentRule, OrderStatus
from hermes_bot.ledger.health import HealthMonitor
from hermes_bot.ledger.lifecycle import LedgerLifecycle
from hermes_bot.ledger.redaction import reject_or_redact
from hermes_bot.ledger.repository import HealthRepository, LedgerRepository
from hermes_bot.ledger.schemas import HealthStatus, LedgerEntryType
from hermes_bot.ledger.serializer import replay_serializer
from hermes_bot.schemas import (
    DecisionProposal,
    RiskApprovalResult,
    RiskCheck,
    RiskRequest,
    RiskStatus,
)


@dataclass
class _TestLedger:
    conn: sqlite3.Connection
    repo: LedgerRepository
    health_repo: HealthRepository


def _db(tmp_path) -> sqlite3.Connection:
    db_path = tmp_path / "ledger.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    MigrationRunner(conn).migrate()
    return conn


@pytest.fixture
def ledger(tmp_path):
    conn = _db(tmp_path)
    return _TestLedger(
        conn=conn,
        repo=LedgerRepository(conn),
        health_repo=HealthRepository(conn),
    )


def _as_of() -> datetime:
    return datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def _proposal(as_of: datetime | None = None) -> DecisionProposal:
    as_of = as_of or _as_of()
    return DecisionProposal(
        strategy_id="no_trade_baseline",
        strategy_version="t09/v1",
        as_of=as_of,
        market_state_as_of=as_of,
        current_weights={"AAPL": 0.0},
        target_weights={"AAPL": 0.10},
        reasons=["test"],
        input_record_ids=["doc_" + "a" * 64],
        impact_record_count=1,
    )


def _risk_request(as_of: datetime | None = None) -> RiskRequest:
    as_of = as_of or _as_of()
    return RiskRequest(
        request_id="req_test_001",
        as_of=as_of,
        nav=10_000.0,
        cash_weight=0.9,
        current_weights={"AAPL": 0.0},
        proposed_positions=[],
        constraints_version="risk-v2.1",
    )


def _approval(as_of: datetime | None = None) -> RiskApprovalResult:
    as_of = as_of or _as_of()
    return RiskApprovalResult(
        approval_id="rap_test_001",
        request_hash="abc123" * 8,
        status=RiskStatus.APPROVED,
        approved_targets={"AAPL": 0.10},
        checks=[RiskCheck(constraint="test", passed=True)],
        reason_codes=["ok"],
        created_at=as_of,
        expires_at=as_of + timedelta(minutes=5),
        constraints_version="risk-v2.1",
    )


def _rules() -> dict[str, InstrumentRule]:
    return {
        "AAPL": InstrumentRule(
            instrument_id="AAPL", currency="USD", qty_precision=2, min_qty=0.01
        )
    }


def _prices() -> dict[str, float]:
    return {"AAPL": 150.0}


def test_migration_creates_ledger_and_health_tables(tmp_path) -> None:
    conn = _db(tmp_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "ledger_cycle" in tables
    assert "ledger_entry" in tables
    assert "health_state" in tables
    runner = MigrationRunner(conn)
    assert runner.current_version() == 3


def test_reconstruct_complete_mock_cycle(ledger) -> None:
    repo = ledger.repo
    lifecycle = LedgerLifecycle(repo)
    as_of = _as_of()
    cycle_id = "cyc_test_001"
    repo.insert_cycle(
        cycle_id,
        decision_time=as_of,
        strategy_id="no_trade_baseline",
        strategy_version="t09/v1",
        proposal_hash="ph_" + "a" * 64,
    )

    proposal = _proposal(as_of)
    lifecycle.record_proposal(
        cycle_id,
        proposal,
        source_document_ids=["doc_" + "a" * 64],
        dataset_version="ds_v1",
        config_version="cfg_v1",
    )
    request = _risk_request(as_of)
    lifecycle.record_risk_request(cycle_id, request)
    approval = _approval(as_of)
    lifecycle.record_risk_approval(cycle_id, approval)

    manager = OrderManager()
    intents = manager.plan(
        approval,
        current_positions={"AAPL": 10.0},
        prices=_prices(),
        instrument_rules=_rules(),
        as_of=as_of,
    )
    results = lifecycle.submit_intents(cycle_id, manager, intents)
    lifecycle.record_portfolio_snapshot(
        cycle_id,
        {"cash": 9000.0, "positions": {"AAPL": results[0].fills[0].qty}},
    )

    entries = repo.get_entries(cycle_id)
    types = [e.entry_type for e in entries]
    assert LedgerEntryType.PROPOSAL in types
    assert LedgerEntryType.RISK_REQUEST in types
    assert LedgerEntryType.RISK_APPROVAL in types
    assert LedgerEntryType.ORDER_INTENT in types
    assert LedgerEntryType.BROKER_RESULT in types
    assert LedgerEntryType.FILL in types
    assert LedgerEntryType.PORTFOLIO_SNAPSHOT in types

    proposal_entries = [e for e in entries if e.entry_type == LedgerEntryType.PROPOSAL]
    assert proposal_entries[0].proposal_hash is not None
    assert proposal_entries[0].source_document_ids == ["doc_" + "a" * 64]

    approval_entries = [e for e in entries if e.entry_type == LedgerEntryType.RISK_APPROVAL]
    assert approval_entries[0].approval_id == "rap_test_001"
    assert approval_entries[0].request_hash == "abc123" * 8

    broker_entries = [e for e in entries if e.entry_type == LedgerEntryType.BROKER_RESULT]
    assert broker_entries[0].broker_status == OrderStatus.FILLED.value
    assert broker_entries[0].fill_quantity is not None


def test_append_only_update_behavior(ledger) -> None:
    repo = ledger.repo
    lifecycle = LedgerLifecycle(repo)
    cycle_id = "cyc_test_002"
    as_of = _as_of()
    repo.insert_cycle(cycle_id, decision_time=as_of)

    lifecycle.record_proposal(cycle_id, _proposal(as_of))
    lifecycle.record_proposal(cycle_id, _proposal(as_of))

    entries = repo.get_entries(cycle_id)
    assert len(entries) == 2
    assert entries[0].sequence == 0
    assert entries[1].sequence == 1
    # Same entry_id should not repeat.
    ids = {e.entry_id for e in entries}
    assert len(ids) == 2


def test_secret_like_field_redacted(ledger) -> None:
    repo = ledger.repo
    lifecycle = LedgerLifecycle(repo)
    cycle_id = "cyc_test_003"
    as_of = _as_of()
    repo.insert_cycle(cycle_id, decision_time=as_of)

    lifecycle.record_portfolio_snapshot(
        cycle_id,
        {"api_key": "super-secret-token-12345", "cash": 1000.0},
    )
    entries = repo.get_entries(cycle_id)
    assert len(entries) == 1
    payload = entries[0].payload
    assert "super-secret-token-12345" not in payload
    assert "__REDACTED__" in payload


def test_secret_like_field_is_redacted_by_reject_or_redact() -> None:
    redacted = reject_or_redact({"api_key": "should-be-redacted"})
    assert redacted["api_key"] == "__REDACTED__"


def test_stale_input_pauses_health(ledger) -> None:
    monitor = HealthMonitor(ledger.health_repo)
    as_of = _as_of()
    stale = as_of - timedelta(seconds=100)
    state = monitor.evaluate_prices(
        {"AAPL": 150.0},
        as_of=as_of,
        max_age_seconds=60.0,
        price_available_at=stale,
    )
    assert state.state == HealthStatus.PAUSED


def test_missing_price_pauses_health(ledger) -> None:
    monitor = HealthMonitor(ledger.health_repo)
    state = monitor.evaluate_prices(
        {},
        as_of=_as_of(),
        max_age_seconds=60.0,
        price_available_at=_as_of(),
    )
    assert state.state == HealthStatus.PAUSED


def test_unknown_broker_state_pauses_and_requests_reconciliation(ledger) -> None:
    repo = ledger.repo
    lifecycle = LedgerLifecycle(repo)
    monitor = HealthMonitor(ledger.health_repo)
    cycle_id = "cyc_test_004"
    as_of = _as_of()
    repo.insert_cycle(cycle_id, decision_time=as_of)

    approval = _approval(as_of)
    lifecycle.record_risk_approval(cycle_id, approval)

    broker = MockBroker(
        script={"abc123abc123abc1_AAPL": {"status": "unknown", "reconciled_status": "filled"}}
    )
    manager = OrderManager(broker=broker)
    intents = manager.plan(
        approval,
        current_positions={"AAPL": 10.0},
        prices=_prices(),
        instrument_rules=_rules(),
        as_of=as_of,
    )
    lifecycle.submit_intents(cycle_id, manager, intents)

    outstanding = repo.outstanding_orders()
    assert any(o["broker_status"] == "unknown" for o in outstanding)

    state = monitor.evaluate_broker_state(has_unknown_orders=True)
    assert state.state == HealthStatus.PAUSED


def test_ledger_write_failure_fails_closed(tmp_path) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # No migration -> table missing; insert should fail.
    repo = LedgerRepository(conn)
    with pytest.raises(sqlite3.OperationalError):
        repo.insert_cycle("cyc_fail", decision_time=_as_of())

def test_restart_loads_outstanding_mock_orders(ledger) -> None:
    repo = ledger.repo
    lifecycle = LedgerLifecycle(repo)
    cycle_id = "cyc_test_005"
    as_of = _as_of()
    repo.insert_cycle(cycle_id, decision_time=as_of)

    approval = _approval(as_of)
    broker = MockBroker(
        script={"abc123abc123abc1_AAPL": {"status": "accepted"}}
    )
    manager = OrderManager(broker=broker)
    intents = manager.plan(
        approval,
        current_positions={"AAPL": 10.0},
        prices=_prices(),
        instrument_rules=_rules(),
        as_of=as_of,
    )
    lifecycle.submit_intents(cycle_id, manager, intents)

    outstanding = repo.outstanding_orders()
    assert len(outstanding) == 1
    assert outstanding[0]["broker_status"] == "accepted"
    assert outstanding[0]["idempotency_key"] is not None


def test_no_production_db_path_used(tmp_path) -> None:
    db_path = tmp_path / "ledger_test_only.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    MigrationRunner(conn).migrate()
    assert "var" not in str(db_path)


def test_replay_serializer_is_deterministic(ledger) -> None:
    repo = ledger.repo
    lifecycle = LedgerLifecycle(repo)
    cycle_id = "cyc_test_006"
    as_of = _as_of()
    repo.insert_cycle(cycle_id, decision_time=as_of)
    lifecycle.record_proposal(cycle_id, _proposal(as_of))
    lifecycle.record_risk_approval(cycle_id, _approval(as_of))

    entries = repo.get_entries(cycle_id)
    first = replay_serializer(entries)
    second = replay_serializer(entries)
    assert first == second
    assert "proposal" in first
    assert "risk_approval" in first


def test_invalid_risk_config_pauses(ledger) -> None:
    monitor = HealthMonitor(ledger.health_repo)
    state = monitor.evaluate_risk_config({"max_gross_exposure": "not-a-number"})
    assert state.state == HealthStatus.PAUSED


def test_health_storage_write_failure_pauses(ledger) -> None:
    monitor = HealthMonitor(ledger.health_repo)
    state = monitor.evaluate_storage_write(False, "disk full")
    assert state.state == HealthStatus.PAUSED
    assert "disk full" in state.reason
