"""Deterministic replay serializer for ledger cycles."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from hermes_bot.ledger.schemas import LedgerEntry


def _canonical_value(value: Any) -> Any:
    """Normalize values for deterministic comparison."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _canonical_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_value(v) for v in value]
    return value


def replay_serializer(entries: list[LedgerEntry]) -> str:
    """Return a canonical JSON representation of a cycle for deterministic replay.

    Timestamps are ISO-8601 strings, keys are sorted and indentation is fixed.
    """
    records: list[dict[str, Any]] = []
    for entry in entries:
        record = {
            "entry_id": entry.entry_id,
            "cycle_id": entry.cycle_id,
            "sequence": entry.sequence,
            "entry_type": entry.entry_type.value,
            "decision_time": entry.decision_time.isoformat() if entry.decision_time else None,
            "source_document_ids": entry.source_document_ids,
            "dataset_version": entry.dataset_version,
            "config_version": entry.config_version,
            "strategy_id": entry.strategy_id,
            "strategy_version": entry.strategy_version,
            "proposal_hash": entry.proposal_hash,
            "approval_id": entry.approval_id,
            "request_hash": entry.request_hash,
            "order_intent_id": entry.order_intent_id,
            "idempotency_key": entry.idempotency_key,
            "broker_order_id": entry.broker_order_id,
            "broker_status": entry.broker_status,
            "instrument_id": entry.instrument_id,
            "fill_quantity": entry.fill_quantity,
            "fill_price": entry.fill_price,
            "fill_fees": entry.fill_fees,
            "portfolio_snapshot": entry.portfolio_snapshot,
            "error_code": entry.error_code,
            "recovery_action": entry.recovery_action,
            "payload": json.loads(entry.payload),
            "created_at": entry.created_at.isoformat(),
        }
        records.append(_canonical_value(record))
    return json.dumps(records, indent=2, sort_keys=True, ensure_ascii=True)
