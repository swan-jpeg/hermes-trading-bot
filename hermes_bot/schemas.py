"""Shared schemas (pydantic) for all modules — the 'language' between layers."""
from __future__ import annotations

from datetime import date, datetime
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
    """Allowed source kinds for provenance records."""

    SPEECH = "speech"
    AUDIO = "audio"
    VIDEO = "video"
    REPORT = "report"
    ALERT = "alert"
    REGIONAL = "regional"
    MARKET = "market"


class Adjustment(StrEnum):
    """Price adjustment semantics for a market bar (§4.2)."""

    RAW = "raw"
    SPLIT_ADJUSTED = "split_adjusted"
    TOTAL_RETURN_ADJUSTED = "total_return_adjusted"


class MarketBar(BaseModel):
    """Canonical OHLCV bar contract (execution package §4.2).

    ``session_date`` is an exchange session date — never a fake UTC timestamp.
    ``interval_start``/``interval_end`` are timezone-aware UTC timestamps when
    known; for a daily bar these may be ``null``.  ``available_at`` is the
    point-in-time at which the bar is safe to use.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    session_date: date
    exchange_mic: str | None = None
    interval_start: datetime | None = None
    interval_end: datetime | None = None
    available_at: datetime
    currency: str = Field(min_length=3, max_length=3)
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    adjustment: Adjustment = Adjustment.RAW
    source_revision: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @property
    def dedup_key(self) -> str:
        """Stable idempotency key for this bar.

        Combines source ID, interval, session date and source revision.
        """
        parts = [self.source_id, self.interval, self.session_date.isoformat()]
        if self.source_revision:
            parts.append(self.source_revision)
        return "|".join(parts)


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


# ---------------------------------------------------------------------------
# Canonical catalog contracts (execution package §4.3 / T04)
# ---------------------------------------------------------------------------
# These models describe the stable entity/instrument/document/event/relation
# foundation stored in the local SQLite catalog.  Decimal prices and weights
# are stored as fractions in named currency; timestamps are timezone-aware UTC.

_ENTITY_ID_RE = r"^ent_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_INS_ID_RE = r"^ins_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_EVT_ID_RE = r"^evt_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_DOC_ID_RE = r"^doc_[0-9a-f]{64}$"
_EDGE_ID_RE = r"^edg_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_DEC_ID_RE = r"^dec_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_SHA256_RE = r"^[0-9a-f]{64}$"


class EntityType(StrEnum):
    """Allowed canonical entity types (§4.3)."""

    COMPANY = "company"
    SECURITY = "security"
    BOND = "bond"
    FUND = "fund"
    INDEX = "index"
    COMMODITY = "commodity"
    CONTRACT = "contract"
    CURRENCY = "currency"
    COUNTRY = "country"
    GOVERNMENT = "government"
    CENTRAL_BANK = "central_bank"
    PERSON = "person"
    ORGANIZATION = "organization"
    INDICATOR = "indicator"
    INDUSTRY = "industry"


class RelationType(StrEnum):
    """Typed temporal economic relations (§4.3, T07)."""

    ISSUER_OF = "issuer_of"
    SUBSIDIARY_OF = "subsidiary_of"
    LOCATED_IN = "located_in"
    MEMBER_OF_SECTOR = "member_of_sector"
    SUPPLIES = "supplies"
    CUSTOMER_OF = "customer_of"
    EXPOSED_TO = "exposed_to"
    POLICY_APPLIES_TO = "policy_applies_to"
    INDICATOR_MEASURES = "indicator_measures"
    CURRENCY_OF = "currency_of"


class EpistemicType(StrEnum):
    """Confidence/status of an extracted claim (§4.3)."""

    CONFIRMED_FACT = "confirmed_fact"
    ALLEGATION = "allegation"
    FORECAST = "forecast"
    OPINION = "opinion"
    UNKNOWN = "unknown"


class StrengthSemantics(StrEnum):
    """What a relation ``strength_value`` means (§4.3)."""

    OBSERVED_ASSOCIATION = "observed_association"
    STRUCTURAL_DEPENDENCY = "structural_dependency"
    CAUSAL_ESTIMATE = "causal_estimate"
    UNKNOWN = "unknown"


class ResolutionStatus(StrEnum):
    """Alias resolution status (§4.3)."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    HISTORICAL = "historical"
    UNRESOLVED = "unresolved"


class InstrumentType(StrEnum):
    """Instrument subtype; kept distinct from asset class (§4.1)."""

    COMMON_STOCK = "common_stock"
    PREFERRED_STOCK = "preferred_stock"
    ETF_SHARE = "etf_share"
    BOND_FIXED = "bond_fixed"
    BOND_FLOATING = "bond_floating"
    OPTION_CALL = "option_call"
    OPTION_PUT = "option_put"
    FUTURE = "future"
    SPOT = "spot"
    CASH = "cash"


class CanonicalEntity(BaseModel):
    """Canonical economic entity (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str = Field(pattern=_ENTITY_ID_RE)
    entity_type: EntityType
    canonical_name: str = Field(min_length=1)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class Instrument(BaseModel):
    """Canonical instrument (§4.1 / §4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str = Field(pattern=_INS_ID_RE)
    issuer_entity_id: str | None = Field(default=None, pattern=_ENTITY_ID_RE)
    asset_class: AssetClass
    exchange_mic: str | None = None
    currency: str = Field(min_length=3, max_length=3)
    instrument_type: InstrumentType
    canonical_name: str = Field(min_length=1)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class Alias(BaseModel):
    """Alias / ticker mapping to a canonical entity (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    entity_id: str = Field(pattern=_ENTITY_ID_RE)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    resolution_status: ResolutionStatus = ResolutionStatus.RESOLVED
    created_at: datetime


class SourceRecord(BaseModel):
    """Provenance record for any raw source object (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_kind: SourceKind
    license_id: str | None = None
    raw_object_path: str | None = None
    published_time: datetime | None = None
    ingested_at: datetime
    available_at: datetime
    metadata: dict[str, object] = Field(default_factory=dict)


class Document(BaseModel):
    """Document metadata; content lives in the raw object store (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(pattern=_DOC_ID_RE)
    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    raw_sha256: str = Field(pattern=_SHA256_RE)
    language: str | None = None
    document_type: str = Field(min_length=1)
    published_time: datetime | None = None
    ingested_at: datetime
    available_at: datetime
    license_id: str | None = None
    raw_object_path: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class Event(BaseModel):
    """Extracted event/claim with evidence references (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(pattern=_EVT_ID_RE)
    event_type: str = Field(min_length=1)
    normalized_claim_text: str = Field(min_length=1)
    stance_direction: str | None = None
    event_time: datetime | None = None
    published_time: datetime | None = None
    available_at: datetime
    epistemic_type: EpistemicType = EpistemicType.UNKNOWN
    novelty: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor_version: str = Field(min_length=1)
    source_document_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    affected_entity_ids: list[str] = Field(default_factory=list)
    uncertainty_fields: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class RelationEdge(BaseModel):
    """Typed temporal relation between two canonical entities (§4.3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    edge_id: str = Field(pattern=_EDGE_ID_RE)
    from_entity_id: str = Field(pattern=_ENTITY_ID_RE)
    relation_type: RelationType
    to_entity_id: str = Field(pattern=_ENTITY_ID_RE)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    observed_at: datetime | None = None
    available_at: datetime
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    strength_value: float | None = None
    strength_semantics: StrengthSemantics = StrengthSemantics.UNKNOWN
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    method_version: str = Field(min_length=1)
    created_at: datetime


class DecisionRecord(BaseModel):
    """Schema-only decision ledger row; populated in T12 (§4.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(pattern=_DEC_ID_RE)
    instrument_id: str = Field(pattern=_INS_ID_RE)
    target_weight: float = Field(ge=-1.0, le=1.0)
    decision_time: datetime
    strategy_id: str | None = None
    strategy_version: str | None = None
    input_record_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    approval_id: str | None = None
    created_at: datetime
