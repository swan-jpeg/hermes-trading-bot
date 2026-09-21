"""Aanbevolen uitbreidingen (optioneel). [[12]]

Implementatie: B2B-vraag & supply-chain bottleneck analyse + regime-detectie.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_bot.signals.textual import ReportSignal


class BottleneckAnalyzer:
    """B2B-vraag & supply-chain bottleneck analyse.

    Vind knelpunten in de keten (bijv. actuators) en geef bottleneck_score.
    Score = concentratie (weinig spelers) + vraagdruk + capaciteitsbeperking.
    """

    def __init__(self) -> None:
        # node -> {customers:set, suppliers:set, demand:float, capacity:float, players:int}
        self.graph: dict[str, dict] = {}

    def add_node(
        self, node: str, demand: float = 0.0, capacity: float = 1.0, players: int = 10
    ) -> None:
        self.graph.setdefault(
            node,
            {
                "customers": set(),
                "suppliers": set(),
                "demand": demand,
                "capacity": capacity,
                "players": players,
            },
        )

    def add_edge(self, supplier: str, customer: str) -> None:
        self.add_node(supplier)
        self.add_node(customer)
        self.graph[supplier]["customers"].add(customer)
        self.graph[customer]["suppliers"].add(supplier)

    def bottleneck_score(self, node: str) -> float:
        """0..1. Hoog = knelpunt (weinig spelers, hoge vraag, lage capaciteit)."""
        n = self.graph.get(node)
        if not n:
            return 0.0
        concentration = 1.0 - min(1.0, n["players"] / 10.0)  # weinig spelers = hoger
        demand_pressure = min(1.0, n["demand"] / max(n["capacity"], 1e-9))
        # Vraagdruk > capaciteit = bottleneck.
        return round(0.4 * concentration + 0.6 * demand_pressure, 4)

    def top_bottlenecks(self, k: int = 5) -> list[tuple[str, float]]:
        scored = [(n, self.bottleneck_score(n)) for n in self.graph]
        return sorted(scored, key=lambda x: -x[1])[:k]

    def ingest(self, reports: list[ReportSignal]) -> dict[str, float]:
        """Verwerk rapporten om vraagsignalen te halen voor bottleneck analyse.

        Args:
            reports: Lijst van ReportSignal objecten

        Returns:
            Dict met bottleneck scores voor elk knooppunt (product/sector)
        """
        # Voor nu een simpele implementatie
        # In een echte implementatie zouden we:
        # 1. Tekstanalyse van rapporten
        # 2. Extractie van vraagsignalen
        # 3. Mapping naar supply-chain knooppunten
        
        # Dummy implementatie
        scores = {}
        for report in reports:
            # Simuleer dat we uit rapporten vraagsignalen halen
            # Voor nu een simpele mapping op basis van keywords
            text = report.body.lower()
            if "supply" in text or "production" in text:
                # Simuleer dat we een product vinden
                scores["manufacturing"] = 0.7
            if "capacity" in text or "shortage" in text:
                scores["supply_chain"] = 0.8
            if "lead time" in text or "delay" in text:
                scores["logistics"] = 0.6
                
        return scores


class RegimeDetector:
    """Classificeert markttoestand: bullish/bearish/highvol/crash."""

    def detect(self, features: dict) -> str:
        """features: vix, credit_spread, yield_curve, momentum.

        Terug: 'bull' | 'bear' | 'highvol' | 'crash'.
        """
        vix = features.get("vix", 20.0)
        spread = features.get("credit_spread", 1.0)
        curve = features.get("yield_curve", 0.0)  # 10y-2y
        momentum = features.get("momentum", 0.0)

        if vix > 40 or spread > 4.0:
            return "crash"
        if vix > 25 or spread > 2.5:
            return "highvol"
        if curve < 0 and momentum < 0:
            return "bear"
        if momentum > 0:
            return "bull"
        return "bear"


class RegionalScorer:
    """Analyseert regionale scores uit nieuws/alerts."""

    def __init__(self) -> None:
        # Regionale scores: veiligheid, tevredenheid, sociale zekerheid, bedrijfseconomisch
        self.scores = {
            "veiligheid": 0.0,
            "tevredenheid": 0.0,
            "sociale_zekerheid": 0.0,
            "bedrijfseconomische_veranderingen": 0.0
        }

    def score_from_reports(self, reports: list[ReportSignal]) -> dict[str, float]:
        """Bereken regionale scores uit rapporten.

        Args:
            reports: Lijst van ReportSignal objecten

        Returns:
            Dict met regionale scores
        """
        # Voor nu een simpele implementatie
        # In een echte implementatie zouden we:
        # 1. Tekstanalyse van rapporten
        # 2. Extractie van regionale informatie
        # 3. Scoring op basis van inhoud
        
        # Dummy implementatie
        total_scores = {
            "veiligheid": 0.0,
            "tevredenheid": 0.0,
            "sociale_zekerheid": 0.0,
            "bedrijfseconomische_veranderingen": 0.0
        }
        
        for report in reports:
            text = report.body.lower()
            
            # Simpele keyword matching
            if "crime" in text or "security" in text:
                total_scores["veiligheid"] += 0.2
            if "happiness" in text or "satisfaction" in text:
                total_scores["tevredenheid"] += 0.2
            if "unemployment" in text or "social" in text:
                total_scores["sociale_zekerheid"] += 0.2
            if "economy" in text or "growth" in text or "inflation" in text:
                total_scores["bedrijfseconomische_veranderingen"] += 0.2
                
        # Normaliseer naar 0-1
        normalized_scores = {}
        for key, value in total_scores.items():
            normalized_scores[key] = min(value, 1.0)
            
        return normalized_scores
