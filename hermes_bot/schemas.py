"""Gedeelde schemas (pydantic) voor alle modulen — de 'taal' tussen de lagen."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    HEDGE = "hedge"
    OTHER = "other"


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
    """Elke signaalbron levert dit minimum."""

    source: SourceKind
    source_name: str
    entity_id: str  # ticker / entiteit
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)


class FusionSignal(BaseModel):
    """Uitgang van het fusiemodel ([[06]]). Kwaliteit/zekerheid/emotie."""

    entity_id: str
    tijdstip: datetime
    kwaliteit: float = Field(ge=0.0, le=1.0)
    zekerheid: float = Field(ge=0.0, le=1.0)
    emotie: dict[str, float]  # bijv. {"positief": ..., "angst": ..., "optimisme": ...}
    source_breakdown: dict[str, float]  # per-bron gewicht/bijdrage


class EntityAnalysis(BaseModel):
    """Uitgang van de fundamentele AI-agent ([[05]])."""

    entity_id: str
    time: datetime
    fundamentals: dict[str, float]  # value, quality, growth, margin, ...
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    regions_exposure: dict[str, float]
    thesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    n_sources: int


class RLRawDecision(BaseModel):
    """Voorstel van de RL-laag. WORDT NOOIT DIRECT UITGEVOERD."""

    entity: str
    action: Action
    intent_to_alloc: float = Field(ge=-1.0, le=1.0)  # -1..+1 portfolio-gewicht
    zekerheid: float = Field(ge=0.0, le=1.0)
    rationale: str
    timestamp: datetime


class RiskApproval(BaseModel):
    """Uitgang van de risico-engine ([[09]]). De enige weg naar executie."""

    approved: bool
    target_alloc: dict[str, float]  # asset -> gewicht
    rejected_reasons: list[str] = Field(default_factory=list)
    adjusted: bool = False
    drawdown_current: float = 0.0
    var_95: float = 0.0
    es_95: float = 0.0


class MonteCarloResult(BaseModel):
    """Uitgang van de Monte Carlo-simulatie ([[08]])."""

    n_paths: int
    horizon: int  # dagen
    percentiles: dict[int, float]  # 5,10,25,50,75,90,95
    var_95: float
    expected_shortfall_95: float
    max_drawdown_distribution: dict[int, float]
    crash_probability: float = Field(ge=0.0, le=1.0)
