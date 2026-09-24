"""AssetContext — the rich input vector for the RL model.

The user specified 6 categories of possible inputs. Not everything is equally
useful or feasible; here is the chosen subset per category, with a clear
reason. Everything is normalized to 0..1 (except where noted), so the RL
model (PPO) gets a stable, scalable observation.

Category 1 — Impact Agent (core): direction, magnitude, confidence,
    probability, duration, directness, novelty, surprise.
Category 2 — Information source: quality, reliability, confidence,
    completeness, cross-source confirmation, freshness, novelty.
Category 3 — Type/content (relevance): company, industry, macro,
    geopolitical, regulatory, technological, supply-chain, demand,
    fundamental.
Category 4 — Market interpretation: sentiment, sentiment-confidence,
    expectation, expectation-surprise, narrative-strength, attention,
    consensus, contrarian.
Category 5 — B2B Bottleneck: demand-growth, supply-scarcity,
    supplier-concentration, bottleneck-strength, pricing-power,
    capacity-constraint, substitutability, barrier-to-entry, confidence.
Category 6 — Time: information-age, impact-decay, expected-duration,
    event-proximity, signal-persistence.
Category 7 — Asset/Business profile: asset-class (one-hot) + sector type,
    region and size class. This lets the model distinguish companies by
    KIND (stock vs bond, tech vs energy, US vs EU, small vs mega) without
    needing a fixed ticker — it generalizes to any entity the impact agent
    points at.

This module is the "language" between the upstream layers (impact/fusion/
bottleneck/webscraping) and the RL model. Each layer fills the fields it
knows; the rest stays at a neutral default (0.5 or 0.0) so the vector is
always complete.

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Key -> profile. Fields are 0..1 and fill category 7. Unknown
# tickers get a neutral profile (asset-class is ALWAYS set)
# from the impact agent; the rest default neutral 0.5/0.0)
#   indexgroep  asset-keywords      sectortype      region      size     volatility
ASSET_PROFILES: dict[str, dict] = {
    # --- Semiconductors / tech hardware ---
    "NVDA": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 1.0,
             "min_supply_chain": 1.0},
    "AMD":  {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 1.0,
             "min_supply_chain": 1.0},
    "INTC": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.7,
             "min_supply_chain": 1.0},
    "TSM":  {"sector_tech": 1.0, "region_asia": 1.0, "size_class": 1.0, "volatility": 0.8,
             "min_supply_chain": 1.0},
    "AVGO": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.7,
             "min_supply_chain": 0.7},
    # --- Software / big tech ---
    "MSFT": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.4,
             "min_supply_chain": 0.0},
    "AAPL": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.5,
             "min_supply_chain": 0.8},
    "GOOGL": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.5,
              "min_supply_chain": 0.3},
    "META": {"sector_tech": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.7,
             "min_supply_chain": 0.2},
    "XLK":  {"sector_tech": 1.0, "region_us": 1.0, "size_class": 0.9, "volatility": 0.5,
             "min_supply_chain": 0.3},
    # --- Consumer / retail ---
    "AMZN": {"sector_consumer": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.6,
             "min_supply_chain": 0.8},
    "WMT":  {"sector_consumer": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.3,
             "min_supply_chain": 0.8},
    "COST": {"sector_consumer": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.3,
             "min_supply_chain": 0.7},
    "XLY":  {"sector_consumer": 1.0, "region_us": 1.0, "size_class": 0.9, "volatility": 0.5,
             "min_supply_chain": 0.5},
    # --- Healthcare / pharma ---
    "JNJ":  {"sector_healthcare": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.2,
             "min_supply_chain": 0.3},
    "PFE":  {"sector_healthcare": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.4,
             "min_supply_chain": 0.4},
    "UNH":  {"sector_healthcare": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.3,
             "min_supply_chain": 0.1},
    "XLV":  {"sector_healthcare": 1.0, "region_us": 1.0, "size_class": 0.9, "volatility": 0.3,
             "min_supply_chain": 0.2},
    # --- Financials ---
    "JPM":  {"sector_financial": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.5,
             "min_supply_chain": 0.0},
    "BAC":  {"sector_financial": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.6,
             "min_supply_chain": 0.0},
    "GS":   {"sector_financial": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.6,
             "min_supply_chain": 0.0},
    "XLF":  {"sector_financial": 1.0, "region_us": 1.0, "size_class": 0.9, "volatility": 0.5,
             "min_supply_chain": 0.0},
    # --- Energy / power / utilities ---
    "XLE":  {"sector_energy": 1.0, "region_us": 1.0, "size_class": 0.85, "volatility": 0.6,
             "min_supply_chain": 0.5},
    "NEE":  {"sector_energy": 1.0, "region_us": 1.0, "size_class": 0.8, "volatility": 0.3,
             "min_supply_chain": 0.3},
    "DUK":  {"sector_energy": 1.0, "region_us": 1.0, "size_class": 0.8, "volatility": 0.2,
             "min_supply_chain": 0.2},
    # --- Materials / battery / supply chain ---
    "ALB":  {"sector_materials": 1.0, "region_us": 1.0, "size_class": 0.6, "volatility": 1.0,
             "min_supply_chain": 1.0},
    "LIT":  {"sector_materials": 1.0, "region_us": 1.0, "size_class": 0.5, "volatility": 0.9,
             "min_supply_chain": 0.9},
    "QS":   {"sector_materials": 1.0, "region_us": 1.0, "size_class": 0.4, "volatility": 1.0,
             "min_supply_chain": 0.9},
    # --- Robots / industrials ---
    "TSLA": {"sector_industrial": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 1.0,
             "min_supply_chain": 0.9},
    "FANUY": {"sector_industrial": 1.0, "region_jp": 1.0, "size_class": 0.8, "volatility": 0.6,
              "min_supply_chain": 0.4},
    # --- Broad equity indices (ETF) ---
    "SPY": {"sector_broad": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.5,
            "min_supply_chain": 0.2},
    "QQQ": {"sector_broad": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.7,
            "min_supply_chain": 0.3},
    "IWM": {"sector_broad": 1.0, "region_us": 1.0, "size_class": 0.3, "volatility": 0.7,
            "min_supply_chain": 0.3},
    "EFA": {"sector_broad": 1.0, "region_eu": 0.8, "region_jp": 0.7, "size_class": 0.9,
            "volatility": 0.5, "min_supply_chain": 0.3},
    # --- Bonds (asset class handles the rest) ---
    "AGG": {"sector_bond": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.3,
            "min_supply_chain": 0.1},
    "TLT": {"sector_bond": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.6,
            "min_supply_chain": 0.0},
    "LQD": {"sector_bond": 1.0, "region_us": 1.0, "size_class": 1.0, "volatility": 0.4,
            "min_supply_chain": 0.1},
    "XLI": {"sector_industrial": 1.0, "region_us": 1.0, "size_class": 0.85, "volatility": 0.5,
            "min_supply_chain": 0.5},
    # Custom tickers used in the impact-agent sector map (demo placeholders).
    "POWCO": {"sector_energy": 1.0, "region_us": 1.0, "size_class": 0.7, "volatility": 0.5,
              "min_supply_chain": 0.5},
    "AIRTECH": {"sector_industrial": 1.0, "region_us": 1.0, "size_class": 0.6,
                "volatility": 0.8, "min_supply_chain": 0.6},
}

@dataclass
class AssetContext:
    """Full, normalized context for one asset (ticker).

    All fields are 0..1 (except where noted). Neutral default is 0.5 for
    "unknown/neutral", 0.0 for "absent".
    """

    # --- Category 1: Impact Agent (core) ---
    impact_direction: float = 0.5      # 0=strongly negative, 1=strongly positive
    impact_magnitude: float = 0.0       # how large the expected impact is
    impact_confidence: float = 0.0      # confidence of the impact agent
    impact_probability: float = 0.0     # chance the impact occurs
    impact_duration: float = 0.5        # how long the impact lasts (0=short,1=long)
    directness: float = 0.0             # 1=directly hit, 0=indirectly
    impact_novelty: float = 0.0         # how new/unexpected
    market_surprise: float = 0.0        # deviation from what was already expected

    # --- Category 2: Information source ---
    source_quality: float = 0.5
    source_reliability: float = 0.5
    info_confidence: float = 0.5
    info_completeness: float = 0.5
    cross_source_confirmation: float = 0.0  # 0=1 source, 1=much confirmation
    info_freshness: float = 0.5             # 1=fresh, 0=old
    info_novelty: float = 0.0

    # --- Category 3: Type/content (relevance) ---
    company_relevance: float = 0.0
    industry_relevance: float = 0.0
    macro_relevance: float = 0.0
    geopolitical_relevance: float = 0.0
    regulatory_relevance: float = 0.0
    technological_relevance: float = 0.0
    supply_chain_relevance: float = 0.0
    demand_relevance: float = 0.0
    fundamental_relevance: float = 0.0

    # --- Category 4: Market interpretation ---
    market_sentiment: float = 0.5        # 0=negative, 1=positive
    sentiment_confidence: float = 0.5
    market_expectation: float = 0.5      # what the market already expected
    expectation_surprise: float = 0.0    # news vs expectation
    narrative_strength: float = 0.0
    market_attention: float = 0.0
    consensus_strength: float = 0.5
    contrarian_strength: float = 0.0

    # --- Category 5: B2B Bottleneck ---
    demand_growth: float = 0.0
    supply_scarcity: float = 0.0
    supplier_concentration: float = 0.0
    bottleneck_strength: float = 0.0
    pricing_power: float = 0.0
    capacity_constraint: float = 0.0
    substitutability: float = 0.5        # 1=hard to substitute
    barrier_to_entry: float = 0.0
    bottleneck_confidence: float = 0.0

    # --- Category 6: Time ---
    information_age: float = 0.0         # 1=just arrived, 0=old
    impact_decay: float = 0.0           # 1=impact still full, 0=played out
    expected_duration: float = 0.5
    event_proximity: float = 0.0         # 1=event now, 0=far away
    signal_persistence: float = 0.0

    # --- Portfolio/risk (existing, from the env) ---
    regime_code: float = 0.0             # 0..3 bull/bear/highvol/crash
    pnl: float = 0.0                     # ongerealiseerde P&L % (-1..+5)
    holding_days: float = 0.0
    allocation: float = 0.0
    var_95: float = 0.0                  # -1..0
    crash_prob: float = 0.0

    # --- Category 7: Asset/Business profile (KIND, generalized) ---
    # Asset-class one-hot (always set from the impact agent).
    asset_is_stock: float = 0.0
    asset_is_bond: float = 0.0
    asset_is_etf: float = 0.0
    asset_is_commodity: float = 0.0
    # Sector type (0..1 per sector; max 1 active).
    sector_tech: float = 0.0
    sector_consumer: float = 0.0
    sector_healthcare: float = 0.0
    sector_financial: float = 0.0
    sector_energy: float = 0.0
    sector_materials: float = 0.0
    sector_industrial: float = 0.0
    # Region.
    region_us: float = 0.0
    region_eu: float = 0.0
    region_jp: float = 0.0
    # Size/volatility profile (0..1).
    size_class: float = 0.5              # 0=small-cap, 1=mega-cap
    volatility: float = 0.5              # 0=low-vol/defensive, 1=high-vol

    # --- Extra metadata (not in the observation, but for logging) ---
    entity: str = ""
    asset_class: str = ""

    # The order of the observation vector (subset per category).
    _OBS_KEYS: tuple[str, ...] = (
        # Category 1 — Impact (core)
        "impact_direction", "impact_magnitude", "impact_confidence",
        "impact_probability", "impact_duration", "directness",
        "impact_novelty", "market_surprise",
        # Category 2 — Information source
        "source_quality", "source_reliability", "info_confidence",
        "info_completeness", "cross_source_confirmation",
        "info_freshness", "info_novelty",
        # Category 3 — Relevance (top-5 chosen)
        "company_relevance", "industry_relevance", "macro_relevance",
        "geopolitical_relevance", "supply_chain_relevance",
        # Category 4 — Market interpretation
        "market_sentiment", "sentiment_confidence", "market_expectation",
        "expectation_surprise", "narrative_strength", "market_attention",
        "consensus_strength", "contrarian_strength",
        # Category 5 — Bottleneck (top-5 chosen)
        "demand_growth", "supply_scarcity", "bottleneck_strength",
        "pricing_power", "capacity_constraint",
        # Category 6 — Time
        "information_age", "impact_decay", "expected_duration",
        "event_proximity", "signal_persistence",
        # Category 7 — Asset/Business profile (KIND)
        "asset_is_stock", "asset_is_bond", "asset_is_etf", "asset_is_commodity",
        "sector_tech", "sector_consumer", "sector_healthcare",
        "sector_financial", "sector_energy", "sector_materials",
        "sector_industrial",
        "region_us", "region_eu", "region_jp",
        "size_class", "volatility",
        # Portfolio/risico
        "regime_code", "pnl", "holding_days", "allocation", "var_95", "crash_prob",
    )

    def observation(self) -> np.ndarray:
        """Build the normalized observation vector (float32)."""
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
        """Fill known fields; ignore unknown ones (so layers can fill separately)."""
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
    """Build an AssetContext from the outputs of the upstream layers.

    Each layer fills only the fields it knows; the rest stays neutral.
    """
    ctx = AssetContext(entity=entity, asset_class=asset_class)

    # Impact agent (category 1).
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

    # Fusion model (category 4 + sentiment).
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

    # B2B bottleneck (category 5).
    if bottleneck:
        ctx.update(
            demand_growth=bottleneck.get("demand_growth", 0.0),
            supply_scarcity=bottleneck.get("supply_scarcity", 0.0),
            bottleneck_strength=bottleneck.get("bottleneck_score", 0.0),
            pricing_power=bottleneck.get("pricing_power", 0.0),
            capacity_constraint=bottleneck.get("capacity_constraint", 0.0),
            bottleneck_confidence=bottleneck.get("confidence", 0.0),
        )

    # Information source (category 2).
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

    # Category 7 — Asset/Business Profile. Always: asset-class one-hot encoding
    # the impact agent (stock/bond/etf/commodity). Further a fixed profile
    # for known tickers (sector/region/size/volume); unknown tickers
    # stay neutral on sector/region and only get the asset class.
    _apply_asset_profile(ctx, asset_class, entity)

    return ctx


def _apply_asset_profile(ctx: AssetContext, asset_class: str, entity: str) -> None:
    """Fill in category 7 (asset class + profile) for this entity."""
    ac = (asset_class or "").lower()
    # Asset-class one-hot (altijd).
    if ac in ("stock", "equity", "share"):
        ctx.asset_is_stock = 1.0
    elif ac in ("bond", "treasury", "credit", "government"):
        ctx.asset_is_bond = 1.0
    elif ac in ("etf", "index", "fund"):
        ctx.asset_is_etf = 1.0
    elif ac in ("commodity", "gold", "oil", "metal"):
        ctx.asset_is_commodity = 1.0
    else:
        # Unknown asset class: default to equity unless clearly a bond.
        ctx.asset_is_stock = 0.5
        ctx.asset_is_bond = 0.0
        ctx.asset_is_etf = 0.0
        ctx.asset_is_commodity = 0.0
    # Fixed profile for known tickers (only the fields we understand).
    profile = ASSET_PROFILES.get((entity or "").upper(), {})
    for k, v in profile.items():
        field = {
            "sector_tech": "sector_tech", "sector_consumer": "sector_consumer",
            "sector_healthcare": "sector_healthcare",
            "sector_financial": "sector_financial",
            "sector_energy": "sector_energy",
            "sector_materials": "sector_materials",
            "sector_industrial": "sector_industrial",
            "region_us": "region_us", "region_eu": "region_eu",
            "region_jp": "region_jp", "region_asia": "region_jp",
            "size_class": "size_class", "volatility": "volatility",
        }.get(k)
        if field and isinstance(v, (int, float)):
            setattr(ctx, field, float(max(0.0, min(1.0, v))))
    # If an ETF/index has no sector profile, leave the sector fields 0.
    # Observable-consistency: one active sector keeps the vector sparse.
