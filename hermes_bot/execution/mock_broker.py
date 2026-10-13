"""Deterministic scripted mock broker for local-only order lifecycle tests."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from hermes_bot.execution.schemas import BrokerResult, Fill, OrderIntent, OrderStatus


class MockBroker:
    """In-memory mock broker with deterministic scripted responses.

    No network calls. Supports accepted, rejected, partial fill, full fill,
    timeout -> unknown, cancel and query. ``unknown`` orders are reconciled
    explicitly (default: resolved to ``filled``) before any retry.
    """

    name = "mock"

    def __init__(
        self,
        script: dict[str, dict[str, Any]] | None = None,
        initial_positions: dict[str, float] | None = None,
    ) -> None:
        self._script: dict[str, dict[str, Any]] = script or {}
        self._orders: dict[str, BrokerResult] = {}
        self._positions: dict[str, float] = dict(initial_positions or {})

    def _order_id(self, intent: OrderIntent) -> str:
        return f"mock_{intent.idempotency_key}"

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _resolve_entry(self, key: str) -> dict[str, Any]:
        return self._script.get(key, {"status": "filled"})

    def submit(self, intent: OrderIntent) -> BrokerResult:
        """Submit an intent and return a deterministic result."""
        key = intent.idempotency_key
        entry = self._resolve_entry(key)
        status = OrderStatus(entry.get("status", "filled"))
        now = self._now()

        if status == OrderStatus.UNKNOWN:
            result = BrokerResult(
                client_id=intent.order_intent_id,
                idempotency_key=key,
                broker_order_id=self._order_id(intent),
                status=OrderStatus.UNKNOWN,
                error_code=entry.get("error_code", "timeout"),
            )
        elif status == OrderStatus.REJECTED:
            result = BrokerResult(
                client_id=intent.order_intent_id,
                idempotency_key=key,
                broker_order_id=None,
                status=OrderStatus.REJECTED,
                error_code=entry.get("error_code", "rejected"),
            )
        elif status == OrderStatus.ACCEPTED:
            result = BrokerResult(
                client_id=intent.order_intent_id,
                idempotency_key=key,
                broker_order_id=self._order_id(intent),
                status=OrderStatus.ACCEPTED,
            )
        elif status == OrderStatus.FILLED:
            price = self._price(entry, intent)
            fill = Fill(
                qty=intent.qty,
                price=price,
                time=now,
                fees=entry.get("fees", 0.0),
            )
            result = BrokerResult(
                client_id=intent.order_intent_id,
                idempotency_key=key,
                broker_order_id=self._order_id(intent),
                status=OrderStatus.FILLED,
                fills=[fill],
            )
            self._update_position(intent, intent.qty)
        elif status == OrderStatus.PARTIALLY_FILLED:
            fill_qty = entry.get("filled_qty", intent.qty / 2.0)
            price = self._price(entry, intent)
            fill = Fill(
                qty=fill_qty,
                price=price,
                time=now,
                fees=entry.get("fees", 0.0),
            )
            result = BrokerResult(
                client_id=intent.order_intent_id,
                idempotency_key=key,
                broker_order_id=self._order_id(intent),
                status=OrderStatus.PARTIALLY_FILLED,
                fills=[fill],
            )
            self._update_position(intent, fill_qty)
        else:
            raise ValueError(f"unsupported scripted status for {key}: {status.value}")

        self._orders[key] = result
        return result

    def _price(self, entry: dict[str, Any], intent: OrderIntent) -> float:
        price = entry.get("price", 100.0)
        if intent.order_type.value == "limit" and intent.limit_price is not None:
            price = intent.limit_price
        return float(price)

    def _update_position(self, intent: OrderIntent, fill_qty: float) -> None:
        sign = 1.0 if intent.side == "buy" else -1.0
        current = self._positions.get(intent.instrument_id, 0.0)
        self._positions[intent.instrument_id] = current + sign * fill_qty

    def query(self, idempotency_key: str) -> BrokerResult | None:
        """Return the latest known broker result for a key."""
        return self._orders.get(idempotency_key)

    def cancel(self, idempotency_key: str) -> BrokerResult | None:
        """Cancel an open order; no-op for terminal states."""
        result = self._orders.get(idempotency_key)
        if result is None:
            return None
        if result.status in {
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        }:
            return result
        cancelled = result.model_copy(update={"status": OrderStatus.CANCELLED})
        self._orders[idempotency_key] = cancelled
        return cancelled

    def reconcile(self, idempotency_key: str) -> BrokerResult | None:
        """Resolve an ``unknown`` order deterministically before retry."""
        result = self._orders.get(idempotency_key)
        if result is None or result.status != OrderStatus.UNKNOWN:
            return result
        entry = self._resolve_entry(idempotency_key)
        resolved_status = OrderStatus(entry.get("reconciled_status", "filled"))
        resolved = result.model_copy(update={"status": resolved_status})
        self._orders[idempotency_key] = resolved
        return resolved

    def get_positions(self) -> dict[str, float]:
        """Current mock positions in instrument units."""
        return dict(self._positions)
