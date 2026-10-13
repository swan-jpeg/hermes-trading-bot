"""Legacy compatibility adapter for the deterministic impact extractor (T08).

The old ``ImpactAgent``/``LLMImpactAgent`` public methods are preserved, but
internally they delegate to the deterministic extractor.  No network calls are
made; the LLM path is explicitly degraded rather than silently falling back.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from hermes_bot.data.catalog import Catalog
from hermes_bot.data.graph import EconomicGraph
from hermes_bot.impact.contracts import (
    Direction,
    ImpactAnalysis,
    ImpactRule,
    TextChunk,
)
from hermes_bot.impact.core import DeterministicImpactExtractor


@dataclass
class ImpactResult:
    """Legacy output shape preserved for callers/tests.

    The numeric ``probability`` field is kept only for structural compatibility;
    it is always zero because the new contract never emits probability.
    """

    event: str
    matched_sectors: list[str] = field(default_factory=list)
    impacted: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    direction: float = 0.5
    magnitude: float = 0.0
    probability: float = 0.0
    duration: float = 0.5
    directness: float = 0.0
    novelty: float = 0.0
    surprise: float = 0.0
    analysis: ImpactAnalysis | None = None


class ImpactAgent:
    """Compatibility wrapper around the deterministic impact extractor."""

    def __init__(
        self,
        sector_map: dict | None = None,
        rules: Sequence[ImpactRule] | None = None,
        catalog: Catalog | None = None,
        graph: EconomicGraph | None = None,
        max_hops: int = 2,
    ) -> None:
        """Create an impact agent.

        ``sector_map`` is accepted for backward compatibility but is ignored;
        rules must be injected via ``rules``.
        """
        self.sector_map = sector_map
        self._extractor = DeterministicImpactExtractor(
            rules=rules,
            catalog=catalog,
            graph=graph,
            max_hops=max_hops,
        )

    def analyze(self, event: str, source: str = "news",
                entity_id: str = "", chunks: list[TextChunk] | None = None) -> ImpactResult:
        """Determine which instruments are hit by this event."""
        analysis = self._extractor.analyze(
            event, source=source, entity_id=entity_id, chunks=chunks
        )
        return _analysis_to_legacy_result(analysis, self._extractor.catalog)

    def to_fusion_inputs(self, result: ImpactResult, sentiment: float = 0.0) -> list[dict]:
        """Convert the impact analysis to fusion inputs per affected entity."""
        if not result.impacted:
            return []
        return [
            {
                "source": "impact",
                "entity_id": item["entity"],
                "asset_class": item.get("asset_class", ""),
                "sentiment": round(float(item.get("sentiment", sentiment)), 4),
                "confidence": round(float(item.get("confidence", result.confidence)), 4),
                "reason": item.get("reason", ""),
            }
            for item in result.impacted
        ]


class LLMImpactAgent(ImpactAgent):
    """Legacy LLM impact agent adapter.

    The network path is disabled.  The agent always delegates to the
    deterministic extractor and marks the result as ``degraded`` with an
    explicit warning.  This replaces the previous silent fallback behavior.
    """

    def __init__(
        self,
        sector_map: dict | None = None,
        rules: Sequence[ImpactRule] | None = None,
        catalog: Catalog | None = None,
        graph: EconomicGraph | None = None,
        max_hops: int = 2,
        api_key: str | None = None,
        model: str = "stepfun/step-3.7-flash:free",
        base_url: str = "https://inference-api.nousresearch.com/v1",
    ) -> None:
        super().__init__(
            sector_map=sector_map,
            rules=rules,
            catalog=catalog,
            graph=graph,
            max_hops=max_hops,
        )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def analyze(self, event: str, source: str = "news",
                entity_id: str = "", chunks: list[TextChunk] | None = None) -> ImpactResult:
        """Analyze using the deterministic extractor; LLM path is disabled."""
        result = super().analyze(event, source, entity_id, chunks=chunks)
        analysis = result.analysis
        if analysis is not None:
            if "LLM interpretation disabled" not in analysis.warnings:
                analysis.warnings.append(
                    "LLM interpretation disabled; deterministic extraction only"
                )
            if analysis.state == "ok":
                analysis.state = "degraded"
        return result


def build_impact_agent(config: dict | None = None) -> ImpactAgent:
    """Factory: build the impact agent (deterministic by default)."""
    cfg = config or {}
    if cfg.get("use_llm"):
        # Network LLM is disabled; return the degraded LLM adapter.
        return LLMImpactAgent(
            api_key=cfg.get("llm_api_key"),
            model=cfg.get("llm_model", "stepfun/step-3.7-flash:free"),
        )
    return ImpactAgent(
        rules=cfg.get("impact_rules"),
        catalog=cfg.get("impact_catalog"),
        graph=cfg.get("impact_graph"),
        max_hops=cfg.get("impact_max_hops", 2),
    )


def _analysis_to_legacy_result(
    analysis: ImpactAnalysis, catalog: Catalog | None
) -> ImpactResult:
    """Map the new deterministic analysis to the legacy ``ImpactResult``."""
    impacted: list[dict] = []
    for record in analysis.impacts:
        entity_id = record.entity_id
        impacted.append({
            "entity": entity_id,
            "asset_class": _asset_class_for_entity(catalog, entity_id),
            "reason": "; ".join(record.reason_codes) or "deterministic impact",
            "confidence": record.subjective_confidence,
        })

    confidence = max((r.subjective_confidence for r in analysis.impacts), default=0.0)
    directions = {r.direction for r in analysis.impacts}
    direction = _direction_to_numeric(_merge_directions(directions))
    has_impacts = bool(analysis.impacts)

    return ImpactResult(
        event=analysis.event.normalized_claim_text,
        matched_sectors=[analysis.event.event_type] + analysis.event.matched_rules,
        impacted=impacted,
        confidence=round(confidence, 4),
        direction=round(direction, 4),
        magnitude=round(confidence, 4),
        probability=0.0,
        duration=0.5 if has_impacts else 0.0,
        directness=1.0 if has_impacts else 0.0,
        novelty=0.3 if has_impacts else 0.0,
        surprise=0.0,
        analysis=analysis,
    )


def _display_id(catalog: Catalog | None, entity_id: str) -> str:
    """Return a human-readable identifier (ticker alias or canonical name)."""
    if catalog is None:
        return entity_id
    ticker = _alias_for_entity(catalog, entity_id, "ticker")
    if ticker:
        return ticker
    entity = catalog.entity.get(entity_id)
    if entity is not None:
        return entity.canonical_name
    instrument = catalog.instrument.get(entity_id)
    if instrument is not None:
        return instrument.canonical_name
    return entity_id


def _asset_class_for_entity(catalog: Catalog | None, entity_id: str) -> str:
    """Best-effort asset class lookup; never invents mappings."""
    if catalog is None:
        return ""
    if entity_id.startswith("ins_"):
        instrument = catalog.instrument.get(entity_id)
        if instrument is not None:
            return instrument.asset_class.value
    entity = catalog.entity.get(entity_id)
    if entity is not None and entity.entity_type.value in {
        "security", "bond", "fund", "index", "commodity", "currency", "contract"
    }:
        return entity.entity_type.value
    return ""


def _alias_for_entity(catalog: Catalog | None, entity_id: str, namespace: str) -> str | None:
    if catalog is None:
        return None
    rows = catalog.conn.execute(
        "SELECT alias FROM alias WHERE entity_id = ? AND namespace = ? LIMIT 1",
        (entity_id, namespace),
    ).fetchall()
    if rows:
        return rows[0]["alias"]
    return None


def _direction_to_numeric(direction: Direction) -> float:
    mapping: dict[Direction, float] = {
        "negative": 0.0,
        "unknown": 0.5,
        "mixed": 0.5,
        "positive": 1.0,
    }
    return mapping.get(direction, 0.5)


def _merge_directions(directions: set[Direction]) -> Direction:
    if len(directions) == 1:
        return next(iter(directions))
    if directions == {"positive", "negative"}:
        return "mixed"
    if "mixed" in directions:
        return "mixed"
    if "unknown" in directions and len(directions) == 2:
        return (directions - {"unknown"}).pop()
    return "unknown"
