"""Lifecycle wiring between decision proposal, risk approval and mock execution."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from hermes_bot.execution.order_manager import OrderManager
from hermes_bot.execution.schemas import BrokerResult, OrderIntent
from hermes_bot.ledger.redaction import reject_or_redact
from hermes_bot.ledger.repository import LedgerRepository
from hermes_bot.ledger.schemas import LedgerEntry, LedgerEntryType
from hermes_bot.schemas import DecisionProposal, RiskApprovalResult, RiskRequest


class LedgerLifecycle:
    """Record each stage of a decision cycle into the append-only ledger.

    The lifecycle helper is intentionally thin: it translates domain objects
    into ``LedgerEntry`` rows and never mutates existing rows.  Broker
    submissions are delegated to the existing ``OrderManager``.
    """

    def __init__(self, ledger: LedgerRepository) -> None:
        self.ledger = ledger

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _next_sequence(self, cycle_id: str) -> int:
        row = self.ledger.conn.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 FROM ledger_entry WHERE cycle_id = ?",
            (cycle_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def _entry(
        self,
        cycle_id: str,
        entry_type: LedgerEntryType,
        payload: Any,
        **fields: Any,
    ) -> LedgerEntry:
        redacted_payload = reject_or_redact(payload)
        sequence = self._next_sequence(cycle_id)
        return LedgerEntry(
            entry_id=f"ent_{uuid.uuid4().hex}",
            cycle_id=cycle_id,
            sequence=sequence,
            entry_type=entry_type,
            payload=json.dumps(redacted_payload, sort_keys=True, default=str),
            created_at=self._now(),
            **fields,
        )

    def record_proposal(
        self,
        cycle_id: str,
        proposal: DecisionProposal,
        source_document_ids: list[str] | None = None,
        dataset_version: str | None = None,
        config_version: str | None = None,
    ) -> LedgerEntry:
        payload = proposal.model_dump(mode="json")
        entry = self._entry(
            cycle_id,
            LedgerEntryType.PROPOSAL,
            payload,
            decision_time=proposal.as_of,
            source_document_ids=source_document_ids or [],
            dataset_version=dataset_version,
            config_version=config_version,
            strategy_id=proposal.strategy_id,
            strategy_version=proposal.strategy_version,
            proposal_hash=self._hash_canonical(payload),
        )
        self.ledger.insert(entry)
        return entry

    def record_risk_request(
        self,
        cycle_id: str,
        request: RiskRequest,
    ) -> LedgerEntry:
        payload = request.model_dump(mode="json")
        entry = self._entry(
            cycle_id,
            LedgerEntryType.RISK_REQUEST,
            payload,
            decision_time=request.as_of,
            config_version=request.constraints_version,
            request_hash=request.request_id,
        )
        self.ledger.insert(entry)
        return entry

    def record_risk_approval(
        self,
        cycle_id: str,
        approval: RiskApprovalResult,
    ) -> LedgerEntry:
        payload = approval.model_dump(mode="json")
        entry = self._entry(
            cycle_id,
            LedgerEntryType.RISK_APPROVAL,
            payload,
            decision_time=approval.created_at,
            approval_id=approval.approval_id,
            request_hash=approval.request_hash,
            config_version=approval.constraints_version,
        )
        self.ledger.insert(entry)
        return entry

    def record_order_intents(
        self,
        cycle_id: str,
        intents: list[OrderIntent],
    ) -> list[LedgerEntry]:
        entries: list[LedgerEntry] = []
        for intent in intents:
            payload = intent.model_dump(mode="json")
            entry = self._entry(
                cycle_id,
                LedgerEntryType.ORDER_INTENT,
                payload,
                decision_time=intent.created_at,
                approval_id=intent.approval_id,
                request_hash=intent.request_hash,
                order_intent_id=intent.order_intent_id,
                idempotency_key=intent.idempotency_key,
                instrument_id=intent.instrument_id,
            )
            self.ledger.insert(entry)
            entries.append(entry)
        return entries

    def record_broker_results(
        self,
        cycle_id: str,
        results: list[BrokerResult],
    ) -> list[LedgerEntry]:
        entries: list[LedgerEntry] = []
        for result in results:
            payload = result.model_dump(mode="json")
            fill_quantity: float | None = None
            fill_price: float | None = None
            fill_fees: float | None = None
            if result.fills:
                fill_quantity = sum(f.qty for f in result.fills)
                fill_price = result.fills[0].price
                fill_fees = sum(f.fees for f in result.fills)
            entry = self._entry(
                cycle_id,
                LedgerEntryType.BROKER_RESULT,
                payload,
                order_intent_id=result.client_id,
                idempotency_key=result.idempotency_key,
                broker_order_id=result.broker_order_id,
                broker_status=result.status.value,
                instrument_id=self._instrument_from_client_id(result.client_id),
                fill_quantity=fill_quantity,
                fill_price=fill_price,
                fill_fees=fill_fees,
            )
            self.ledger.insert(entry)
            entries.append(entry)
            for fill in result.fills:
                self.record_fill(cycle_id, result, fill)
        return entries

    def record_fill(
        self,
        cycle_id: str,
        result: BrokerResult,
        fill: Any,
    ) -> LedgerEntry:
        payload = {
            "client_id": result.client_id,
            "idempotency_key": result.idempotency_key,
            "broker_order_id": result.broker_order_id,
            "fill": fill.model_dump(mode="json"),
        }
        entry = self._entry(
            cycle_id,
            LedgerEntryType.FILL,
            payload,
            order_intent_id=result.client_id,
            idempotency_key=result.idempotency_key,
            broker_order_id=result.broker_order_id,
            instrument_id=self._instrument_from_client_id(result.client_id),
            fill_quantity=fill.qty,
            fill_price=fill.price,
            fill_fees=fill.fees,
        )
        self.ledger.insert(entry)
        return entry

    def record_portfolio_snapshot(
        self,
        cycle_id: str,
        snapshot: dict[str, Any],
    ) -> LedgerEntry:
        entry = self._entry(
            cycle_id,
            LedgerEntryType.PORTFOLIO_SNAPSHOT,
            snapshot,
            portfolio_snapshot=json.dumps(snapshot, sort_keys=True, default=str),
        )
        self.ledger.insert(entry)
        return entry

    def record_error(
        self,
        cycle_id: str,
        error_code: str,
        recovery_action: str,
        payload: Any | None = None,
    ) -> LedgerEntry:
        entry = self._entry(
            cycle_id,
            LedgerEntryType.ERROR,
            payload or {"error_code": error_code},
            error_code=error_code,
            recovery_action=recovery_action,
        )
        self.ledger.insert(entry)
        return entry

    def record_recovery(
        self,
        cycle_id: str,
        recovery_action: str,
        payload: Any | None = None,
    ) -> LedgerEntry:
        entry = self._entry(
            cycle_id,
            LedgerEntryType.RECOVERY,
            payload or {"recovery_action": recovery_action},
            recovery_action=recovery_action,
        )
        self.ledger.insert(entry)
        return entry

    def submit_intents(
        self,
        cycle_id: str,
        manager: OrderManager,
        intents: list[OrderIntent],
    ) -> list[BrokerResult]:
        """Submit intents through the order manager and record both sides."""
        self.record_order_intents(cycle_id, intents)
        results = manager.submit_mock(intents)
        self.record_broker_results(cycle_id, results)
        return results

    def _hash_canonical(self, data: dict[str, Any]) -> str:
        """Stable hash for canonical payloads."""
        import hashlib

        canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _instrument_from_client_id(client_id: str) -> str | None:
        parts = client_id.split("_")
        # intent_<instrument>_<approval_id>
        if len(parts) >= 3 and parts[0] == "intent":
            return parts[1]
        return None
