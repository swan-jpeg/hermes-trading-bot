"""Execution-layer schemas — order intent, broker result and instrument rule."""
from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class OrderStatus(StrEnum):
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class Fill(BaseModel):
    """One fill slice for a broker result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qty: float = Field(gt=0.0)
    price: float = Field(gt=0.0)
    time: datetime
    fees: float = Field(default=0.0, ge=0.0)

    @field_validator("qty", "price", "fees")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class OrderIntent(BaseModel):
    """Canonical order intent derived from a RiskApprovalResult (§4.6).

    Immutable. The ``idempotency_key`` is the broker-facing deduplication key;
    ``order_intent_id`` is a stable logical identifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    order_intent_id: str = Field(min_length=1)
    approval_id: str = Field(min_length=1)
    request_hash: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    side: OrderSide
    qty: float = Field(gt=0.0)
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    currency: str = Field(min_length=3, max_length=3)
    time_in_force: TimeInForce = TimeInForce.DAY
    idempotency_key: str = Field(min_length=1)
    created_at: datetime
    expires_at: datetime

    @field_validator("qty")
    @classmethod
    def _qty_finite_positive(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("qty must be finite and positive")
        return v

    @field_validator("limit_price")
    @classmethod
    def _limit_price_valid(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v) or v <= 0:
            raise ValueError("limit_price must be finite and positive")
        return v


class BrokerResult(BaseModel):
    """Broker-facing order lifecycle event (§4.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    broker_order_id: str | None = None
    status: OrderStatus
    fills: list[Fill] = Field(default_factory=list)
    error_code: str | None = None


class InstrumentRule(BaseModel):
    """Per-instrument execution metadata: precision, currency, minimums."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    qty_precision: int = Field(ge=0)
    price_precision: int = Field(default=2, ge=0)
    min_qty: float = Field(default=0.0, ge=0.0)
    allow_fractional: bool = True


class OrderManagerError(Exception):
    """Base error for order-manager violations."""


class ApprovalRejected(OrderManagerError):
    """Raised when an approval is missing, expired or mismatched."""


class DuplicateIntent(OrderManagerError):
    """Raised when an idempotency key is reused with a different payload."""
