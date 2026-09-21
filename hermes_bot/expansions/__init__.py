"""Aanbevolen uitbreidingen (optioneel). [[12]]"""
from __future__ import annotations


class BottleneckAnalyzer:
    """B2B-vraag & supply-chain bottleneck analyse.

    Vind knelpunten in de keten (bijv. actuators) en geef bottleneck_score.
    """

    def __init__(self) -> None:
        # nodes: bedrijf/branche; edges: klant-leverancier.
        self.graph: dict[str, dict] = {}

    def add_edge(self, supplier: str, customer: str) -> None:
        self.graph.setdefault(supplier, {}).setdefault("customers", set()).add(customer)
        self.graph.setdefault(customer, {}).setdefault("suppliers", set()).add(supplier)

    def bottleneck_score(self, node: str) -> float:
        """Concentratie + vraagdruk + capaciteitsbeperking -> 0..1."""
        raise NotImplementedError


class RegimeDetector:
    """Classificeert markttoestand: bullish/bearish/highvol/crash."""

    def detect(self, features: dict) -> str:
        """features: VIX, credit spread, yield curve, momentum.
        Terug: toestand ('bull'|'bear'|'highvol'|'crash').
        """
        raise NotImplementedError
