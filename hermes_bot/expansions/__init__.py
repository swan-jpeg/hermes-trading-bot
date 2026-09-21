"""Aanbevolen uitbreidingen (optioneel). [[12]]

Implementatie: B2B-vraag & supply-chain bottleneck analyse + regime-detectie.
"""
from __future__ import annotations


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
