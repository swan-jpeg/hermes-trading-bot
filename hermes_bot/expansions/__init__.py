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

    def to_fusion_input(self, sentiment: float = 0.0) -> dict:
        """Convert the top bottleneck into a fusion input.

        Dit is de brug: bottleneck-analyse -> fusion model. De bottleneck-score
        wordt vertaald naar een 'sentiment'-achtige richting: hoge bottleneck =
        hoge vraagdruk = positief voor leveranciers in de keten (historisch),
        maar ook verhoogd leveringsrisico. We gebruiken een netto richting.
        """
        tops = self.top_bottlenecks(1)
        if not tops:
            return {}
        node, score = tops[0]
        return {
            "source": "bottleneck",
            "entity_id": node,
            "sentiment": round(score * 0.5 + sentiment * 0.5, 4),
            "confidence": round(0.3 + 0.7 * score, 4),  # hoge score = hogere zekerheid
            "bottleneck_score": round(score, 4),
            "node": node,
        }

    def ingest(self, reports: list[ReportSignal]) -> dict[str, float]:
        """Process reports to extract demand signals for bottleneck analysis.

        Args:
            reports: Lijst van ReportSignal objecten

        Returns:
            Dict met bottleneck scores voor elk knooppunt (product/sector)
        """
        # For now a simple implementation
        # In a real implementation we would:
        # 1. Text analysis of reports
        # 2. Extraction of demand signals
        # 3. Mapping to supply-chain nodes
        
        # Dummy implementatie
        scores = {}
        for report in reports:
            # Simulate extracting demand signals from reports
            # For now a simple mapping based on keywords
            text = report.body.lower()
            if "supply" in text or "production" in text:
                # Simulate finding a product
                scores["manufacturing"] = 0.7
            if "capacity" in text or "shortage" in text:
                scores["supply_chain"] = 0.8
            if "lead time" in text or "delay" in text:
                scores["logistics"] = 0.6
                
        return scores


class BottleneckAgent:
    """Rates companies based on their position in the bottleneck chain.

    Dit is de 'agent' die het bottleneck concept uitwerkt: uit webrapporten
    worden vraagsignalen gehaald, gekoppeld aan supply-chain knooppunten, en
    per bedrijf wordt een rating berekend. Die rating gaat als fusion-input
    naar het fusion model.

    entity->bottlenodes mapping: welke bottleneck-knooppunten blootstellen.
    """

    def __init__(self, analyzer: BottleneckAnalyzer | None = None) -> None:
        self.analyzer = analyzer or BottleneckAnalyzer()
        # Bedrijf -> relevante bottleneck-knooppunten (blootstelling).
        self.entity_exposure: dict[str, list[str]] = {}
        self.ratings: dict[str, float] = {}

    def set_exposure(self, entity: str, nodes: list[str]) -> None:
        self.entity_exposure[entity] = nodes

    def rate_entities(self, market_sentiment: float = 0.0) -> dict[str, float]:
        """Rating per company = average bottleneck score of its nodes.

        Hoge rating = levert op een knelpunt (vraagdruk > capaciteit) ->
        verhoogde omzet-potentie, maar ook leveringsrisico. De rating is
        richtinggevend voor het fusion model.
        """
        ratings = {}
        for entity, nodes in self.entity_exposure.items():
            if not nodes:
                ratings[entity] = 0.0
                continue
            scores = [self.analyzer.bottleneck_score(n) for n in nodes]
            avg = sum(scores) / len(scores)
            ratings[entity] = round(avg, 4)
        self.ratings = ratings
        return ratings

    def to_fusion_inputs(self, market_sentiment: float = 0.0) -> list[dict]:
        """Convert per-company ratings into fusion inputs.

        Elke input is {source: 'bottleneck', entity_id, sentiment, confidence,
        bottleneck_score}. Dit voedt het fusion model per bedrijf.
        """
        self.rate_entities(market_sentiment)
        inputs = []
        for entity, rating in self.ratings.items():
            if rating <= 0:
                continue
            inputs.append({
                "source": "bottleneck",
                "entity_id": entity,
                "sentiment": round(rating * 0.5 + market_sentiment * 0.5, 4),
                "confidence": round(0.3 + 0.7 * rating, 4),
                "bottleneck_score": round(rating, 4),
                "node": self.entity_exposure.get(entity, [""])[0],
            })
        return inputs

    def run_pipeline(
        self,
        reports: list[dict],
        entity_map: dict[str, list[str]],
        market_sentiment: float = 0.0,
    ) -> list[dict]:
        """Volledige keten: reports -> bottleneck-analyse -> bedrijfs-ratings -> fusie-input.

        reports: ruwe webitems {headline, body}.
        entity_map: {bedrijf: [bottleneck-nodes]}.
        """
        # 1. Build the supply-chain graph from reports (demand signals).
        for r in reports:
            text = f"{r.get('headline', '')} {r.get('body', '')}".lower()
            for node in ["semiconductor", "actuators", "power", "logistics"]:
                if node in text:
                    self.analyzer.add_node(node, demand=0.8, capacity=0.4, players=3)
        # 2. Link companies to nodes.
        for entity, nodes in entity_map.items():
            self.set_exposure(entity, nodes)
        # 3. Ratings -> fusie-inputs.
        return self.to_fusion_inputs(market_sentiment)


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
    """Analyses regional scores from news/alerts."""

    def __init__(self) -> None:
        # Regionale scores: veiligheid, tevredenheid, sociale certainty, bedrijfseconomisch
        self.scores = {
            "veiligheid": 0.0,
            "tevredenheid": 0.0,
            "sociale_zekerheid": 0.0,
            "bedrijfseconomische_veranderingen": 0.0
        }

    def score_from_reports(self, reports: list[ReportSignal]) -> dict[str, float]:
        """Compute regional scores from reports.

        Args:
            reports: Lijst van ReportSignal objecten

        Returns:
            Dict met regionale scores
        """
        # For now a simple implementation
        # In a real implementation we would:
        # 1. Text analysis of reports
        # 2. Extraction of regional information
        # 3. Scoring based on content
        
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
                
        # Normalize to 0-1
        normalized_scores = {}
        for key, value in total_scores.items():
            normalized_scores[key] = min(value, 1.0)
            
        return normalized_scores
