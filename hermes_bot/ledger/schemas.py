"""Ledger schemas: append-only entries and health state."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LedgerEntryType(StrEnum):
    """Event types written to the append-only ledger."""

    PROPOSAL = "proposal"
    RISK_REQUEST = "risk_request"
    RISK_APPROVAL = "risk_approval"
    ORDER_INTENT = "order_intent"
    BROKER_RESULT = "broker_result"
    FILL = "fill"
    PORTFOLIO_SNAPSHOT = "portfolio_snapshot"
    ERROR = "error"
    RECOVERY = "recovery"
    HEALTH = "health"


class HealthStatus(StrEnum):
    """Allowed health states."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    PAUSED = "PAUSED"


class LedgerEntry(BaseModel):
    """One append-only ledger row.

    Only the fields relevant to the entry type are populated; the rest are
    optional. The ``payload`` stores the canonical redacted JSON of the event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str = Field(min_length=1)
    cycle_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    entry_type: LedgerEntryType
    decision_time: datetime | None = None
    source_document_ids: list[str] = Field(default_factory=list)
    dataset_version: str | None = None
    config_version: str | None = None
    strategy_id: str | None = None
    strategy_version: str | None = None
    proposal_hash: str | None = None
    approval_id: str | None = None
    request_hash: str | None = None
    order_intent_id: str | None = None
    idempotency_key: str | None = None
    broker_order_id: str | None = None
    broker_status: str | None = None
    instrument_id: str | None = None
    fill_quantity: float | None = None
    fill_price: float | None = None
    fill_fees: float | None = None
    portfolio_snapshot: str | None = None
    error_code: str | None = None
    recovery_action: str | None = None
    payload: str = Field(min_length=1)
    created_at: datetime


class HealthState(BaseModel):
    """Current system health snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: HealthStatus
    reason: str = ""
    updated_at: datetime
