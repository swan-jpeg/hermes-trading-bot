"""Approval-bound order manager: plan intents from RiskApproval, submit to mock broker."""
from __future__ import annotations

import math
from datetime import datetime

from hermes_bot.execution.mock_broker import MockBroker
from hermes_bot.execution.schemas import (
    ApprovalRejected,
    BrokerResult,
    DuplicateIntent,
    InstrumentRule,
    OrderIntent,
    OrderSide,
    OrderStatus,
    TimeInForce,
)
from hermes_bot.schemas import RiskApprovalResult


class OrderManager:
    """Derives mock order intents only from a current immutable RiskApproval.

    - ``plan`` turns an approved target map + current positions into bounded
      order intents and registers the approval.
    - ``submit_mock`` sends intents to the in-memory MockBroker, enforces
      idempotency and reconciles ``unknown`` orders before retry.
    - ``set_kill_switch`` pauses all new submissions fail-closed.

    No real broker calls are made.
    """

    def __init__(self, broker: MockBroker | None = None) -> None:
        self.broker = broker if broker is not None else MockBroker()
        self._idempotency: dict[str, OrderIntent] = {}
        self._results: dict[str, BrokerResult] = {}
        self._approvals: dict[str, RiskApprovalResult] = {}
        self._kill_switch = False

    @property
    def kill_switch(self) -> bool:
        return self._kill_switch

    def set_kill_switch(self, active: bool) -> None:
        """Activate/deactivate the kill switch. When active, new orders are rejected."""
        self._kill_switch = active

    def _intent_payload(self, intent: OrderIntent) -> str:
        """Canonical JSON representation for payload comparison."""
        return intent.model_dump_json()

    def _register_approval(self, approval: RiskApprovalResult) -> None:
        self._approvals[approval.approval_id] = approval

    def _validate_approval_live(self, approval: RiskApprovalResult, as_of: datetime) -> None:
        if not approval.executable:
            raise ApprovalRejected(
                f"approval {approval.approval_id} not executable: {approval.status.value}"
            )
        if as_of >= approval.expires_at:
            raise ApprovalRejected(
                f"approval {approval.approval_id} expired at {approval.expires_at}"
            )
        if not approval.request_hash or len(approval.request_hash) < 16:
            raise ApprovalRejected(f"approval {approval.approval_id} has invalid request_hash")
        if not approval.constraints_version:
            raise ApprovalRejected(
                f"approval {approval.approval_id} has missing constraints_version"
            )

    def plan(
        self,
        approval: RiskApprovalResult,
        current_positions: dict[str, float],
        prices: dict[str, float],
        instrument_rules: dict[str, InstrumentRule],
        as_of: datetime,
    ) -> list[OrderIntent]:
        """Derive order intents from an approved target map and current holdings.

        ``current_positions`` maps instrument_id to current quantity in units.
        ``prices`` maps instrument_id to current price in the instrument currency.
        NAV is derived as the sum of ``qty * price`` for the supplied positions;
        target quantities are ``target_weight * nav / price``.

        Raises ``ApprovalRejected`` for missing/stale approval, missing price,
        invalid precision, non-finite or non-positive quantity, or short-selling.
        """
        self._validate_approval_live(approval, as_of)
        self._register_approval(approval)

        # Seed the broker with current ledger positions so it tracks resulting
        # positions rather than just executed deltas.
        self.broker._positions = dict(current_positions)

        nav = 0.0
        for instrument_id, qty in current_positions.items():
            price = prices.get(instrument_id)
            if price is not None and math.isfinite(price) and price > 0 and math.isfinite(qty):
                nav += qty * price

        intents: list[OrderIntent] = []
        created_at = as_of
        expires_at = approval.expires_at

        all_ids = set(approval.approved_targets) | set(current_positions)
        for instrument_id in sorted(all_ids):
            rule = instrument_rules.get(instrument_id)
            if rule is None:
                raise ApprovalRejected(f"missing instrument rule for {instrument_id}")

            target_weight = approval.approved_targets.get(instrument_id, 0.0)
            current_qty = current_positions.get(instrument_id, 0.0)

            price = prices.get(instrument_id)
            if price is None:
                raise ApprovalRejected(f"missing price for {instrument_id}")
            if not math.isfinite(price) or price <= 0:
                raise ApprovalRejected(f"invalid or stale price for {instrument_id}: {price}")

            if nav > 0:
                target_qty = target_weight * nav / price
            else:
                target_qty = 0.0
            delta_qty = target_qty - current_qty

            if abs(delta_qty) <= rule.min_qty + 1e-12:
                continue

            side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
            qty = abs(delta_qty)

            # Precision.
            precision = rule.qty_precision
            if not math.isfinite(qty):
                raise ApprovalRejected(f"non-finite quantity for {instrument_id}")
            qty = round(qty, precision)
            if qty <= 0:
                continue

            # Long-only MVP: reject short-sell attempts.
            if side == OrderSide.SELL and current_qty <= 0:
                raise ApprovalRejected(f"short selling rejected for {instrument_id}")

            intent_id = f"intent_{instrument_id}_{approval.approval_id}"
            idempotency_key = f"{approval.request_hash[:16]}_{instrument_id}"

            intent = OrderIntent(
                order_intent_id=intent_id,
                approval_id=approval.approval_id,
                request_hash=approval.request_hash,
                instrument_id=instrument_id,
                side=side,
                qty=qty,
                currency=rule.currency,
                time_in_force=TimeInForce.DAY,
                idempotency_key=idempotency_key,
                created_at=created_at,
                expires_at=expires_at,
            )
            intents.append(intent)

        return intents

    def submit_mock(self, intents: list[OrderIntent]) -> list[BrokerResult]:
        """Submit intents to the mock broker with idempotency and reconciliation.

        Raises ``ApprovalRejected`` when kill switch is active, the approval is
        unknown/expired or the intent request_hash mismatches. Raises
        ``DuplicateIntent`` when an idempotency key is reused with a different
        payload.
        """
        if self._kill_switch:
            raise ApprovalRejected("kill switch active: new order submissions paused")

        results: list[BrokerResult] = []
        for intent in intents:
            approval = self._approvals.get(intent.approval_id)
            if approval is None:
                raise ApprovalRejected(f"unknown approval {intent.approval_id}")
            if intent.request_hash != approval.request_hash:
                raise ApprovalRejected(
                    f"intent request_hash {intent.request_hash} does not match approval"
                )

            payload = self._intent_payload(intent)
            previous_intent = self._idempotency.get(intent.idempotency_key)
            if previous_intent is not None:
                if self._intent_payload(previous_intent) != payload:
                    raise DuplicateIntent(
                        f"idempotency key {intent.idempotency_key} reused with different payload"
                    )
                cached = self._results.get(intent.idempotency_key)
                if cached is not None:
                    if cached.status == OrderStatus.UNKNOWN:
                        reconciled = self.broker.reconcile(intent.idempotency_key)
                        if reconciled is not None:
                            self._results[intent.idempotency_key] = reconciled
                            results.append(reconciled)
                            continue
                    results.append(cached)
                    continue

            result = self.broker.submit(intent)
            self._idempotency[intent.idempotency_key] = intent
            self._results[intent.idempotency_key] = result
            results.append(result)

        return results

    def reconcile_positions(self, current_positions: dict[str, float]) -> dict[str, float]:
        """Return signed discrepancies: broker_position - expected_position.

        A non-empty result indicates positions that need reconciliation.
        """
        broker_positions = self.broker.get_positions()
        mismatches: dict[str, float] = {}
        for instrument_id in set(current_positions) | set(broker_positions):
            expected = current_positions.get(instrument_id, 0.0)
            actual = broker_positions.get(instrument_id, 0.0)
            if abs(expected - actual) > 1e-9:
                mismatches[instrument_id] = actual - expected
        return mismatches

    def cancel(self, idempotency_key: str) -> BrokerResult | None:
        """Cancel an open mock order by idempotency key."""
        return self.broker.cancel(idempotency_key)

    def query(self, idempotency_key: str) -> BrokerResult | None:
        """Query the latest mock broker result for an idempotency key."""
        return self.broker.query(idempotency_key)

    def reconcile_unknown(self, idempotency_key: str) -> BrokerResult | None:
        """Explicitly reconcile one unknown mock order."""
        return self.broker.reconcile(idempotency_key)
