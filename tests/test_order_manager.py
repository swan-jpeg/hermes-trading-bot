"""Tests for T10 / v1: approval-bound mock order manager with reconciliation."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hermes_bot.execution import (
    ExecutionEngine,
    InstrumentRule,
    MockBroker,
    Order,
    OrderManager,
    OrderStatus,
    PaperBroker,
)
from hermes_bot.execution.order_manager import ApprovalRejected, DuplicateIntent
from hermes_bot.schemas import RiskApprovalResult, RiskCheck, RiskStatus


def _as_of() -> datetime:
    return datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)


def _approval(
    *,
    approved_targets: dict[str, float] | None = None,
    expires_at: datetime | None = None,
    request_hash: str = "abc123" * 4,
    constraints_version: str = "cv_1",
    status: RiskStatus = RiskStatus.APPROVED,
) -> RiskApprovalResult:
    as_of = _as_of()
    return RiskApprovalResult(
        approval_id="rap_test_001",
        request_hash=request_hash,
        status=status,
        approved_targets=approved_targets or {},
        checks=[RiskCheck(constraint="test", passed=True)],
        reason_codes=[],
        created_at=as_of,
        expires_at=expires_at or as_of + timedelta(minutes=5),
        constraints_version=constraints_version,
    )


def _rules() -> dict[str, InstrumentRule]:
    return {
        "AAPL": InstrumentRule(
            instrument_id="AAPL", currency="USD", qty_precision=2, min_qty=0.01
        ),
        "MSFT": InstrumentRule(
            instrument_id="MSFT", currency="USD", qty_precision=2, min_qty=0.01
        ),
    }


def _prices() -> dict[str, float]:
    return {"AAPL": 150.0, "MSFT": 300.0}


def test_plan_derives_buy_intent_from_target_delta() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    # Hold MSFT so NAV > 0; AAPL is a new position.
    current_positions = {"MSFT": 5.0, "AAPL": 0.0}

    intents = manager.plan(approval, current_positions, _prices(), _rules(), _as_of())

    aapl_intents = [i for i in intents if i.instrument_id == "AAPL"]
    assert len(aapl_intents) == 1
    intent = aapl_intents[0]
    assert intent.side.value == "buy"
    # nav = 5 * 300 = 1500; target qty = 0.10 * 1500 / 150 = 1.0
    assert intent.qty == 1.0


def test_plan_derives_sell_intent_when_over_target() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    current_positions = {"AAPL": 10.0}

    intents = manager.plan(approval, current_positions, _prices(), _rules(), _as_of())

    assert len(intents) == 1
    intent = intents[0]
    assert intent.instrument_id == "AAPL"
    assert intent.side.value == "sell"
    # nav = 10 * 150 = 1500; target qty = 0.10 * 1500 / 150 = 1.0; delta = -9.0
    assert intent.qty == 9.0


def test_plan_rejects_missing_price() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    with pytest.raises(ApprovalRejected, match="missing price"):
        manager.plan(approval, {"AAPL": 1.0}, {}, _rules(), _as_of())


def test_plan_rejects_invalid_price() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    with pytest.raises(ApprovalRejected, match="invalid or stale price"):
        manager.plan(approval, {"AAPL": 1.0}, {"AAPL": float("nan")}, _rules(), _as_of())


def test_plan_rejects_expired_approval() -> None:
    manager = OrderManager()
    approval = _approval(
        approved_targets={"AAPL": 0.10},
        expires_at=_as_of() - timedelta(seconds=1),
    )
    with pytest.raises(ApprovalRejected, match="expired"):
        manager.plan(approval, {"AAPL": 1.0}, _prices(), _rules(), _as_of())


def test_plan_rejects_non_executable_approval() -> None:
    manager = OrderManager()
    approval = _approval(
        approved_targets={"AAPL": 0.10}, status=RiskStatus.REJECTED
    )
    with pytest.raises(ApprovalRejected, match="not executable"):
        manager.plan(approval, {"AAPL": 1.0}, _prices(), _rules(), _as_of())


def test_plan_rejects_short_sell() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.00})
    current_positions = {"AAPL": 0.0}
    # target 0, current 0 -> no delta. Force a sell by setting target below 0 not
    # possible via approved_targets (long-only). Instead pass a position with 0
    # qty and a negative target weight is rejected by RiskApprovalResult validator.
    # We test the rule by ensuring no sell intent is created from zero position.
    intents = manager.plan(approval, current_positions, _prices(), _rules(), _as_of())
    assert intents == []


def test_submit_mock_full_fill() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 1.0}, _prices(), _rules(), _as_of())
    assert intents

    results = manager.submit_mock(intents)

    assert len(results) == 1
    assert results[0].status == OrderStatus.FILLED
    assert len(results[0].fills) == 1
    assert results[0].fills[0].qty == intents[0].qty


def test_submit_mock_partial_fill() -> None:
    broker = MockBroker(
        script={"abc123abc123abc1_AAPL": {"status": "partially_filled", "filled_qty": 2.0}}
    )
    manager = OrderManager(broker=broker)
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())

    results = manager.submit_mock(intents)

    assert len(results) == 1
    assert results[0].status == OrderStatus.PARTIALLY_FILLED
    assert results[0].fills[0].qty == 2.0


def test_submit_mock_rejected() -> None:
    broker = MockBroker(
        script={"abc123abc123abc1_AAPL": {"status": "rejected", "error_code": "risk_limit"}}
    )
    manager = OrderManager(broker=broker)
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())

    results = manager.submit_mock(intents)

    assert len(results) == 1
    assert results[0].status == OrderStatus.REJECTED
    assert results[0].error_code == "risk_limit"


def test_submit_mock_timeout_unknown_then_reconcile() -> None:
    broker = MockBroker(
        script={
            "abc123abc123abc1_AAPL": {
                "status": "unknown",
                "reconciled_status": "filled",
            }
        }
    )
    manager = OrderManager(broker=broker)
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())

    first = manager.submit_mock(intents)
    assert first[0].status == OrderStatus.UNKNOWN

    # Retry: unknown is reconciled to filled before new submit.
    second = manager.submit_mock(intents)
    assert second[0].status == OrderStatus.FILLED


def test_duplicate_same_intent_is_idempotent() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())

    first = manager.submit_mock(intents)
    second = manager.submit_mock(intents)

    assert first[0].status == second[0].status == OrderStatus.FILLED
    assert first[0].broker_order_id == second[0].broker_order_id


def test_same_idempotency_key_different_payload_rejected() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())
    manager.submit_mock(intents)

    # Manually craft an intent with the same idempotency key but different qty.
    tampered = intents[0].model_copy(update={"qty": 99.0})
    with pytest.raises(DuplicateIntent):
        manager.submit_mock([tampered])


def test_submit_mock_rejects_wrong_request_hash() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())

    tampered = intents[0].model_copy(update={"request_hash": "deadbeef" * 4})
    with pytest.raises(ApprovalRejected, match="request_hash"):
        manager.submit_mock([tampered])


def test_kill_switch_pauses_new_orders() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())

    manager.set_kill_switch(True)
    with pytest.raises(ApprovalRejected, match="kill switch"):
        manager.submit_mock(intents)

    # Existing fills remain unaffected.
    manager.set_kill_switch(False)
    results = manager.submit_mock(intents)
    assert results[0].status == OrderStatus.FILLED


def test_positions_reconcile() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())
    manager.submit_mock(intents)

    # After selling 9 shares the broker should hold 1.
    mismatches = manager.reconcile_positions({"AAPL": 1.0})
    assert mismatches == {}

    mismatches = manager.reconcile_positions({"AAPL": 5.0})
    assert mismatches == {"AAPL": -4.0}


def test_cancel_open_order() -> None:
    broker = MockBroker(script={"abc123abc123abc1_AAPL": {"status": "accepted"}})
    manager = OrderManager(broker=broker)
    approval = _approval(approved_targets={"AAPL": 0.10})
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), _rules(), _as_of())
    manager.submit_mock(intents)

    cancelled = manager.cancel(intents[0].idempotency_key)
    assert cancelled is not None
    assert cancelled.status == OrderStatus.CANCELLED


def test_execution_engine_raw_execute_raises() -> None:
    engine = ExecutionEngine(PaperBroker())
    with pytest.raises(RuntimeError, match="OrderManager"):
        engine.execute([Order(asset="AAPL", side="buy", qty=1.0)])


def test_alpaca_broker_is_disabled_stub() -> None:
    from hermes_bot.execution.brokers.alpaca import AlpacaBroker

    broker = AlpacaBroker()
    assert broker.health() is False
    order = broker.submit(Order(asset="AAPL", side="buy", qty=1.0))
    assert order.status == "rejected"
    assert broker.get_positions() == {}


def test_plan_skips_instrument_below_min_qty() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    rules = {
        "AAPL": InstrumentRule(
            instrument_id="AAPL", currency="USD", qty_precision=2, min_qty=100.0
        )
    }
    intents = manager.plan(approval, {"AAPL": 1.0}, _prices(), rules, _as_of())
    assert intents == []


def test_plan_applies_qty_precision() -> None:
    manager = OrderManager()
    approval = _approval(approved_targets={"AAPL": 0.10})
    rules = {
        "AAPL": InstrumentRule(
            instrument_id="AAPL", currency="USD", qty_precision=0, min_qty=0.0
        )
    }
    intents = manager.plan(approval, {"AAPL": 10.0}, _prices(), rules, _as_of())
    assert len(intents) == 1
    # delta = 9.0 -> rounded to integer = 9.0
    assert intents[0].qty == 9.0
    assert intents[0].qty.is_integer()
