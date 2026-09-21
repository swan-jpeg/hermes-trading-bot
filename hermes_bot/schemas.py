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


class ExitReason(StrEnum):
    """Waarom een positie gesloten/verkleind wordt — maakt winst nemen expliciet."""

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
    """Elke signaalbron levert dit minimum."""

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


class Position(BaseModel):
    """Een open positie. Geeft de beslissingslaag prijs-bewustzijn zodat
    winst genomen en verlies beperkt kan worden ([[07]]/[[09]])."""

    entity: str
    qty: float = Field(gt=0.0)
    entry_price: float = Field(gt=0.0)
    entry_time: datetime
    take_profit_pct: float = Field(default=0.15, ge=0.0)  # +15% default
    stop_loss_pct: float = Field(default=0.08, ge=0.0)  # -8% default
    trailing_stop_pct: float = Field(default=0.05, ge=0.0)  # -5% vanaf hoogtepunt

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """On-gerealiseerde winst/verlies in % t.o.v. entry."""
        if current_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def exit_reason_at(self, current_price: float, peak_price: float) -> ExitReason:
        """Bepaal of de positie gesloten moet worden o.b.v. prijsregels."""
        pnl = self.unrealized_pnl_pct(current_price)
        if pnl >= self.take_profit_pct:
            return ExitReason.TAKE_PROFIT
        if pnl <= -self.stop_loss_pct:
            return ExitReason.STOP_LOSS
        # Trailing stop: vanaf het hoogtepunt sinds entry.
        if peak_price > self.entry_price:
            drawdown_from_peak = (peak_price - current_price) / peak_price
            if drawdown_from_peak >= self.trailing_stop_pct:
                return ExitReason.TRAILING_STOP
        return ExitReason.NONE


class RLRawDecision(BaseModel):
    """Voorstel van de RL-laag. WORDT NOOIT DIRECT UITGEVOERD."""

    entity: str
    action: Action
    intent_to_alloc: float = Field(ge=-1.0, le=1.0)  # -1..+1 portfolio-gewicht
    zekerheid: float = Field(ge=0.0, le=1.0)
    rationale: str
    timestamp: datetime
    exit_reason: ExitReason = ExitReason.NONE  # gevuld bij SELL/HEDGE


class RiskApproval(BaseModel):
    """Uitgang van de risico-engine ([[09]]). De enige weg naar executie."""

    approved: bool
    target_alloc: dict[str, float]  # asset -> gewicht
    rejected_reasons: list[str] = Field(default_factory=list)
    adjusted: bool = False
    drawdown_current: float = 0.0
    var_95: float = 0.0
    es_95: float = 0.0
    crash_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class MonteCarloResult(BaseModel):
    """Uitgang van de Monte Carlo-simulatie ([[08]])."""

    n_paths: int
    horizon: int  # dagen
    percentiles: dict[int, float]  # 5,10,25,50,75,90,95
    var_95: float
    expected_shortfall_95: float
    max_drawdown_distribution: dict[int, float]
    crash_probability: float = Field(ge=0.0, le=1.0)