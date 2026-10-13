"""Shared schemas (pydantic) for all modules — the 'language' between layers."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    HEDGE = "hedge"
    OTHER = "other"


class ExitReason(StrEnum):
    """Why a position is closed/reduced — makes taking profit explicit."""

    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SENTIMENT = "sentiment"
    RISK_OFF = "risk_off"
    NONE = "none"


class AssetClass(StrEnum):
    STOCK = "stock"
    ETF = "etf"
    OPTION = "option"
    BOND = "bond"
    CASH = "cash"


class SourceKind(StrEnum):
    SPEECH = "speech"
    AUDIO = "audio"
    VIDEO = "video"
    REPORT = "report"
    ALERT = "alert"
    REGIONAL = "regional"
    MARKET = "market"


class BaseSignal(BaseModel):
    """Every signal source provides this minimum."""

    source: SourceKind
    source_name: str
    entity_id: str  # ticker / entiteit
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)


class AlertSignal(BaseSignal):
    source: SourceKind = SourceKind.ALERT
    kind: str = "news"  # news | point_loop
    headline: str = ""
    body: str = ""
    sentiment: float = 0.0  # -1..1
    credibility: float = 0.5  # 0..1 bronweging


class FusionSignal(BaseModel):
    """Output of the fusion model ([[06]]). Quality/certainty/emotion."""

    entity_id: str
    tijdstip: datetime
    kwaliteit: float = Field(ge=0.0, le=1.0)
    zekerheid: float = Field(ge=0.0, le=1.0)
    emotie: dict[str, float]  # bijv. {"positief": ..., "angst": ..., "optimisme": ...}
    source_breakdown: dict[str, float]  # per-bron gewicht/bijdrage


class EntityAnalysis(BaseModel):
    """Output of the fundamental AI agent ([[05]])."""

    entity_id: str
    time: datetime
    fundamentals: dict[str, float]  # value, quality, growth, margin, ...
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    regions_exposure: dict[str, float]
    thesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    n_sources: int


class Position(BaseModel):
    """An open position. Gives the decision layer price awareness so that
    winst genomen en verlies beperkt kan worden ([[07]]/[[09]])."""

    entity: str
    qty: float = Field(gt=0.0)
    entry_price: float = Field(gt=0.0)
    entry_time: datetime
    take_profit_pct: float = Field(default=0.15, ge=0.0)  # +15% default
    stop_loss_pct: float = Field(default=0.08, ge=0.0)  # -8% default
    trailing_stop_pct: float = Field(default=0.05, ge=0.0)  # -5% vanaf hoogtepunt

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized profit/loss in % vs entry."""
        if current_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def exit_reason_at(self, current_price: float, peak_price: float) -> ExitReason:
        """Determine whether the position should be closed based on price rules."""
        pnl = self.unrealized_pnl_pct(current_price)
        if pnl >= self.take_profit_pct:
            return ExitReason.TAKE_PROFIT
        if pnl <= -self.stop_loss_pct:
            return ExitReason.STOP_LOSS
        # Trailing stop: from the high since entry.
        if peak_price > self.entry_price:
            drawdown_from_peak = (peak_price - current_price) / peak_price
            if drawdown_from_peak >= self.trailing_stop_pct:
                return ExitReason.TRAILING_STOP
        return ExitReason.NONE


class RLRawDecision(BaseModel):
    """Proposal from the RL layer. NEVER EXECUTED DIRECTLY."""

    entity: str
    action: Action
    intent_to_alloc: float = Field(ge=-1.0, le=1.0)  # -1..+1 portfolio-gewicht
    zekerheid: float = Field(ge=0.0, le=1.0)
    rationale: str
    timestamp: datetime
    exit_reason: ExitReason = ExitReason.NONE  # gevuld bij SELL/HEDGE


class RiskApproval(BaseModel):
    """Output of the risk engine ([[09]]). The only path to execution."""

    approved: bool
    target_alloc: dict[str, float]  # asset -> gewicht
    rejected_reasons: list[str] = Field(default_factory=list)
    adjusted: bool = False
    drawdown_current: float = 0.0
    var_95: float = 0.0
    es_95: float = 0.0
    crash_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class MonteCarloResult(BaseModel):
    """Output of the Monte Carlo simulation ([[08]])."""

    n_paths: int
    horizon: int  # dagen
    percentiles: dict[int, float]  # 5,10,25,50,75,90,95
    var_95: float
    expected_shortfall_95: float
    max_drawdown_distribution: dict[int, float]
    crash_probability: float = Field(ge=0.0, le=1.0)
    # Padding model that was used in the simulation (bootstrap | gbm | jump_diffusion | regime).
    model: str = "bootstrap"
    # Crash-scenario's forceren ([[08]] punt 3): scenario-naam -> portefeuille-rendement.
    scenario_table: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Whole-portfolio risk approval contract (execution package, contracts §4.6)
# ---------------------------------------------------------------------------
# These models implement the deterministic, portfolio-level v2.1 risk approval.
# They are long-only MVP: target weights are NAV fractions in [0, 1]. They do
# NOT by themselves authorize execution — a later task wires order intents.


class RiskStatus(StrEnum):
    """Outcome of a whole-portfolio risk approval."""

    APPROVED = "approved"
    REDUCED = "reduced"
    REJECTED = "rejected"
    PAUSED = "paused"


class RiskTargetPosition(BaseModel):
    """One proposed target position inside a portfolio risk request (§4.6).

    ``target_weight`` is a NAV fraction; long-only MVP requires ``[0, 1]``.
    Price freshness is proven either explicitly via ``is_price_fresh`` or
    derived from ``price_available_at`` vs the request ``as_of``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str
    target_weight: float
    price_available_at: datetime | None = None
    price_source_id: str | None = None
    # Explicit freshness flag: True = fresh, False = stale. None = derive from
    # price_available_at. A required price that is neither explicit-fresh nor
    # backed by a timestamp is treated as missing.
    is_price_fresh: bool | None = None
    sector_id: str | None = None
    decision_time: datetime | None = None
    strategy_id: str | None = None
    strategy_version: str | None = None
    input_record_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class RiskRequest(BaseModel):
    """Complete target portfolio for one decision cycle (§4.6).

    ``current_weights`` and ``proposed_positions`` are NAV fractions. All
    timestamps must be timezone-aware UTC. ``constraints_version`` identifies
    the immutable config under which the approval is granted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    as_of: datetime
    nav: float
    cash_weight: float
    current_weights: dict[str, float] = Field(default_factory=dict)
    proposed_positions: list[RiskTargetPosition] = Field(default_factory=list)
    constraints_version: str
    # Optional per-source max age in seconds. For OFFLINE_MOCK fixtures this
    # is left None: freshness is then only fixture available_at <= as_of.
    max_stale_seconds: float | None = None
    # Optional known-instrument universe; an ID outside it is "unknown".
    universe_ids: list[str] | None = None
    # Long-only MVP: short/margin requests are unsupported and rejected.
    allow_short: bool = False
    allow_margin: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class RiskCheck(BaseModel):
    """A single per-constraint check inside an approval result (§4.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint: str
    passed: bool
    observed: float | None = None
    limit: float | None = None
    reason_code: str = ""


class RiskApprovalResult(BaseModel):
    """Immutable whole-portfolio approval (§4.6).

    The only object that may later seed order intents. ``approved_targets`` is
    the complete approved target map (every proposed instrument, zeros for
    unapproved/rejected ones). ``request_hash`` is the canonical SHA-256 of
    the serialized request including the constraints version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    approval_id: str
    request_hash: str
    status: RiskStatus
    approved_targets: dict[str, float]
    checks: list[RiskCheck] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime
    expires_at: datetime
    constraints_version: str

    @property
    def executable(self) -> bool:
        """True only when the approval authorizes a target map."""
        return self.status in (RiskStatus.APPROVED, RiskStatus.REDUCED)
