"""Impact Agent — koppelt gebeurtenissen aan beïnvloede instrumenten.

De ontbrekende schakel in de keten: bepaalt WELKE instrumenten (stocks,
obligaties, ETF's) geraakt worden door een gebeurtenis (speech, kwartaalrapport,
overheidsuitgave, nieuws). Dit is een PARALLELLE stap vóór het fusion-model:

    webscraping (gebeurtenis)
        ↓
    IMPACT-AGENT  ← bepaalt: "dit raakt NVDA, TSLA, AGG"
        ↓
    per beïnvloede entiteit: fusion-output (sentiment/zekerheid/kwaliteit)
        ↓
    RL-model (per entiteit: welk instrument + gewicht + zekerheid)
        ↓
    risk engine + Monte Carlo → portfolio → executie

Zonder deze agent is het RL-model "blind" — het weet niet welke stocks geraakt
worden door een speech of rapport. Deze agent levert die koppeling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Sector/onderwerp -> beïnvloede instrumenten (tickers + assetklasse).
# Uitbreidbaar: voeg keywords toe per sector.
_SECTOR_MAP: dict[str, dict] = {
    "semiconductor": {
        "keywords": ["chip", "semiconductor", "halfgeleider", "nvidia", "tsmc",
                     "ai chip", "gpu", "datacenter"],
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO"],
        "asset_class": "stock",
    },
    "actuators": {
        "keywords": ["actuator", "robot", "robotics", "automation", "motion"],
        "tickers": ["TSLA", "AIRTECH", "FANUY"],
        "asset_class": "stock",
    },
    "power": {
        "keywords": ["power", "energy", "electricity", "grid", "utility", "energie"],
        "tickers": ["POWCO", "NEE", "DUK", "XLE"],
        "asset_class": "stock",
    },
    "battery": {
        "keywords": ["battery", "lithium", "ev", "electric vehicle", "accu"],
        "tickers": ["TSLA", "QS", "ALB", "LIT"],
        "asset_class": "stock",
    },
    "government": {
        "keywords": ["overheid", "government", "uitgave", "spending", "stimulus",
                     "begroting", "budget", "infrastructuur", "infrastructure"],
        "tickers": ["AGG", "TLT", "LQD", "XLI"],
        "asset_class": "bond",
    },
    "macro": {
        "keywords": ["inflation", "rente", "interest rate", "cpi", "werkloosheid",
                     "unemployment", "gdp", "economie", "economy"],
        "tickers": ["SPY", "QQQ", "IWM", "EFA", "AGG"],
        "asset_class": "etf",
    },
    "consumer": {
        "keywords": ["consumer", "retail", "winkel", "verkoop", "sales", "besteding"],
        "tickers": ["AMZN", "WMT", "COST", "XLY"],
        "asset_class": "stock",
    },
    "healthcare": {
        "keywords": ["health", "pharma", "ziekenhuis", "medisch", "drug", "vaccin"],
        "tickers": ["JNJ", "PFE", "UNH", "XLV"],
        "asset_class": "stock",
    },
    "finance": {
        "keywords": ["bank", "finance", "financieel", "krediet", "credit", "loan"],
        "tickers": ["JPM", "BAC", "GS", "XLF"],
        "asset_class": "stock",
    },
    "tech": {
        "keywords": ["tech", "software", "cloud", "ai", "kunstmatige intelligentie",
                     "internet", "platform"],
        "tickers": ["MSFT", "AAPL", "GOOGL", "META", "XLK"],
        "asset_class": "stock",
    },
}


@dataclass
class ImpactResult:
    """Output van de impact-agent: welke instrumenten worden geraakt."""

    event: str
    matched_sectors: list[str] = field(default_factory=list)
    impacted: list[dict] = field(default_factory=list)  # [{entity, asset_class, reason}]
    confidence: float = 0.0


class ImpactAgent:
    """Koppelt een gebeurtenis aan beïnvloede instrumenten via keyword-matching."""

    def __init__(self, sector_map: dict | None = None) -> None:
        self.sector_map = sector_map or _SECTOR_MAP

    def analyze(self, event: str, source: str = "news",
               entity_id: str = "") -> ImpactResult:
        """Bepaal welke instrumenten door deze gebeurtenis worden geraakt.

        event: de tekst van de gebeurtenis (speech, rapport, overheidsuitgave).
        source: speech | report | alert | government.
        entity_id: optionele entiteit (bv. "trump") die ook als hint dient.
        """
        text = (event + " " + entity_id).lower()
        # Speech van een wereldleider (trump, biden, putin, ...) -> macro-impact.
        if source == "speech" and any(
            leader in entity_id.lower()
            for leader in ("trump", "biden", "putin", "leader", "president", "minister")
        ):
            text += " inflation economy government"
        matched: list[str] = []
        impacted: list[dict] = []
        for sector, info in self.sector_map.items():
            if any(kw in text for kw in info["keywords"]):
                matched.append(sector)
                for ticker in info["tickers"]:
                    impacted.append({
                        "entity": ticker,
                        "asset_class": info["asset_class"],
                        "reason": f"{sector} ({source})",
                    })
        # Dedupliceer op entity (behoud eerste).
        seen: set[str] = set()
        unique: list[dict] = []
        for item in impacted:
            if item["entity"] not in seen:
                seen.add(item["entity"])
                unique.append(item)
        confidence = min(1.0, 0.3 + 0.2 * len(matched)) if matched else 0.0
        return ImpactResult(
            event=event,
            matched_sectors=matched,
            impacted=unique,
            confidence=round(confidence, 4),
        )

    def to_fusion_inputs(self, result: ImpactResult, sentiment: float = 0.0) -> list[dict]:
        """Zet de impact-analyse om naar fusion-inputs per beïnvloede entiteit.

        Elke beïnvloede entiteit krijgt een eigen fusion-input, zodat het
        fusion-model + RL-model per instrument kunnen beslissen.
        """
        if not result.impacted:
            return []
        return [
            {
                "source": "impact",
                "entity_id": item["entity"],
                "asset_class": item["asset_class"],
                "sentiment": round(sentiment, 4),
                "confidence": result.confidence,
                "reason": item["reason"],
            }
            for item in result.impacted
        ]


def build_impact_agent(config: dict | None = None) -> ImpactAgent:
    """Factory: bouw de impact-agent (uitbreidbaar via config)."""
    return ImpactAgent()
