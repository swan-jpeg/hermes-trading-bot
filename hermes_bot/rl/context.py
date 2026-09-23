"""AssetContext — de rijke input-vector voor het RL-model.

De gebruiker specificeerde 6 categorieën van mogelijke inputs. Niet alles is
even nuttig of haalbaar; hier is de gekozen subset per categorie, met een
duidelijke reden. Alles is genormaliseerd naar 0..1 (behalve waar anders
aangegeven), zodat het RL-model (PPO) een stabiele, schaalbare observatie
krijgt.

Categorie 1 — Impact Agent (kern): direction, magnitude, confidence,
    probability, duration, directness, novelty, surprise.
Categorie 2 — Informatiebron: quality, reliability, confidence,
    completeness, cross-source confirmation, freshness, novelty.
Categorie 3 — Type/inhoud (relevance): company, industry, macro,
    geopolitical, regulatory, technological, supply-chain, demand,
    fundamental.
Categorie 4 — Marktinterpretatie: sentiment, sentiment-confidence,
    expectation, expectation-surprise, narrative-strength, attention,
    consensus, contrarian.
Categorie 5 — B2B Bottleneck: demand-growth, supply-scarcity,
    supplier-concentration, bottleneck-strength, pricing-power,
    capacity-constraint, substitutability, barrier-to-entry, confidence.
Categorie 6 — Tijd: information-age, impact-decay, expected-duration,
    event-proximity, signal-persistence.

Deze module is de "taal" tussen de upstream-lagen (impact/fusion/bottleneck/
webscraping) en het RL-model. Elke laag vult de velden die hij kent; de rest
blijft op een neutrale default (0.5 of 0.0) zodat de vector altijd volledig is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AssetContext:
    """Volledige, genormaliseerde context voor één asset (ticker).

    Alle velden zijn 0..1 (behalve waar anders vermeld). Neutrale default is
    0.5 voor "onbekend/neutraal", 0.0 voor "afwezig".
    """

    # --- Categorie 1: Impact Agent (kern) ---
    impact_direction: float = 0.5      # 0=sterk negatief, 1=sterk positief
    impact_magnitude: float = 0.0       # hoe groot de verwachte impact is
    impact_confidence: float = 0.0      # zekerheid van de impact agent
    impact_probability: float = 0.0     # kans dat de impact optreedt
    impact_duration: float = 0.5        # hoe lang de impact blijft (0=kort,1=lang)
    directness: float = 0.0             # 1=direct geraakt, 0=indirect
    impact_novelty: float = 0.0         # hoe nieuw/onverwacht
    market_surprise: float = 0.0        # afwijking van wat al verwacht werd

    # --- Categorie 2: Informatiebron ---
    source_quality: float = 0.5
    source_reliability: float = 0.5
    info_confidence: float = 0.5
    info_completeness: float = 0.5
    cross_source_confirmation: float = 0.0  # 0=1 bron, 1=veel bevestiging
    info_freshness: float = 0.5             # 1=vers, 0=oud
    info_novelty: float = 0.0

    # --- Categorie 3: Type/inhoud (relevance) ---
    company_relevance: float = 0.0
    industry_relevance: float = 0.0
    macro_relevance: float = 0.0
    geopolitical_relevance: float = 0.0
    regulatory_relevance: float = 0.0
    technological_relevance: float = 0.0
    supply_chain_relevance: float = 0.0
    demand_relevance: float = 0.0
    fundamental_relevance: float = 0.0

    # --- Categorie 4: Marktinterpretatie ---
    market_sentiment: float = 0.5        # 0=negatief, 1=positief
    sentiment_confidence: float = 0.5
    market_expectation: float = 0.5      # wat de markt al verwachtte
    expectation_surprise: float = 0.0    # nieuws vs verwachting
    narrative_strength: float = 0.0
    market_attention: float = 0.0
    consensus_strength: float = 0.5
    contrarian_strength: float = 0.0

    # --- Categorie 5: B2B Bottleneck ---
    demand_growth: float = 0.0
    supply_scarcity: float = 0.0
    supplier_concentration: float = 0.0
    bottleneck_strength: float = 0.0
    pricing_power: float = 0.0
    capacity_constraint: float = 0.0
    substitutability: float = 0.5        # 1=moeilijk vervangbaar
    barrier_to_entry: float = 0.0
    bottleneck_confidence: float = 0.0

    # --- Categorie 6: Tijd ---
    information_age: float = 0.0         # 1=net binnen, 0=oud
    impact_decay: float = 0.0           # 1=impact nog vol, 0=uitgewerkt
    expected_duration: float = 0.5
    event_proximity: float = 0.0         # 1=gebeurtenis nu, 0=ver weg
    signal_persistence: float = 0.0

    # --- Portfolio/risico (bestaande, uit de env) ---
    regime_code: float = 0.0             # 0..3 bull/bear/highvol/crash
    pnl: float = 0.0                     # ongerealiseerde P&L % (-1..+5)
    holding_days: float = 0.0
    allocation: float = 0.0
    var_95: float = 0.0                  # -1..0
    crash_prob: float = 0.0

    # --- Extra metadata (niet in de observatie, wel voor logging) ---
    entity: str = ""
    asset_class: str = ""

    # De volgorde van de observatie-vector (subset per categorie).
    _OBS_KEYS: tuple[str, ...] = (
        # Categorie 1 — Impact (kern)
        "impact_direction", "impact_magnitude", "impact_confidence",
        "impact_probability", "impact_duration", "directness",
        "impact_novelty", "market_surprise",
        # Categorie 2 — Informatiebron
        "source_quality", "source_reliability", "info_confidence",
        "info_completeness", "cross_source_confirmation",
        "info_freshness", "info_novelty",
        # Categorie 3 — Relevance (top-5 gekozen)
        "company_relevance", "industry_relevance", "macro_relevance",
        "geopolitical_relevance", "supply_chain_relevance",
        # Categorie 4 — Marktinterpretatie
        "market_sentiment", "sentiment_confidence", "market_expectation",
        "expectation_surprise", "narrative_strength", "market_attention",
        "consensus_strength", "contrarian_strength",
        # Categorie 5 — Bottleneck (top-5 gekozen)
        "demand_growth", "supply_scarcity", "bottleneck_strength",
        "pricing_power", "capacity_constraint",
        # Categorie 6 — Tijd
        "information_age", "impact_decay", "expected_duration",
        "event_proximity", "signal_persistence",
        # Portfolio/risico
        "regime_code", "pnl", "holding_days", "allocation", "var_95", "crash_prob",
    )

    def observation(self) -> np.ndarray:
        """Bouw de genormaliseerde observatie-vector (float32)."""
        return np.array([float(getattr(self, k)) for k in self._OBS_KEYS],
                        dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return len(self._OBS_KEYS)

    def to_dict(self) -> dict:
        d = {k: float(getattr(self, k)) for k in self._OBS_KEYS}
        d["entity"] = self.entity
        d["asset_class"] = self.asset_class
        return d

    def update(self, **kwargs) -> AssetContext:
        """Vul bekende velden; negeer onbekende (zodat lagen los kunnen vullen)."""
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                try:
                    setattr(self, k, float(v))
                except (TypeError, ValueError):
                    pass
        return self


def build_asset_context(
    impact: dict | None = None,
    fusion: dict | None = None,
    bottleneck: dict | None = None,
    source: dict | None = None,
    entity: str = "",
    asset_class: str = "",
) -> AssetContext:
    """Bouw een AssetContext uit de outputs van de upstream-lagen.

    Elke laag vult alleen de velden die hij kent; de rest blijft neutraal.
    """
    ctx = AssetContext(entity=entity, asset_class=asset_class)

    # Impact agent (categorie 1).
    if impact:
        ctx.update(
            impact_direction=impact.get("direction", 0.5),
            impact_magnitude=impact.get("magnitude", 0.0),
            impact_confidence=impact.get("confidence", 0.0),
            impact_probability=impact.get("probability", 0.0),
            impact_duration=impact.get("duration", 0.5),
            directness=impact.get("directness", 0.0),
            impact_novelty=impact.get("novelty", 0.0),
            market_surprise=impact.get("surprise", 0.0),
        )

    # Fusion model (categorie 4 + sentiment).
    if fusion:
        sent = fusion.get("emotie", {}).get("sentiment", 0.0)
        ctx.update(
            market_sentiment=float(np.clip((sent + 1.0) / 2.0, 0, 1)),
            sentiment_confidence=fusion.get("zekerheid", 0.5),
            market_expectation=fusion.get("market_expectation", 0.5),
            expectation_surprise=fusion.get("expectation_surprise", 0.0),
            narrative_strength=fusion.get("narrative_strength", 0.0),
            market_attention=fusion.get("market_attention", 0.0),
            consensus_strength=fusion.get("consensus_strength", 0.5),
            contrarian_strength=fusion.get("contrarian_strength", 0.0),
        )

    # B2B bottleneck (categorie 5).
    if bottleneck:
        ctx.update(
            demand_growth=bottleneck.get("demand_growth", 0.0),
            supply_scarcity=bottleneck.get("supply_scarcity", 0.0),
            bottleneck_strength=bottleneck.get("bottleneck_score", 0.0),
            pricing_power=bottleneck.get("pricing_power", 0.0),
            capacity_constraint=bottleneck.get("capacity_constraint", 0.0),
            bottleneck_confidence=bottleneck.get("confidence", 0.0),
        )

    # Informatiebron (categorie 2).
    if source:
        ctx.update(
            source_quality=source.get("quality", 0.5),
            source_reliability=source.get("reliability", 0.5),
            info_confidence=source.get("confidence", 0.5),
            info_completeness=source.get("completeness", 0.5),
            cross_source_confirmation=source.get("cross_source_confirmation", 0.0),
            info_freshness=source.get("freshness", 0.5),
            info_novelty=source.get("novelty", 0.0),
        )

    return ctx
