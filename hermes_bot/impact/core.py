"""Deterministic, evidence-linked impact extractor (T08).

The extractor is intentionally narrow: substring rule matching, exact catalog
alias resolution, and bounded typed-graph traversal.  It never sends network
requests, never invents ticker mappings, and never produces an
``estimated_return``.
"""
from __future__ import annotations

import re
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from hermes_bot.data.catalog import Catalog
from hermes_bot.data.graph import EconomicGraph, GraphPath
from hermes_bot.impact.contracts import (
    AnalysisState,
    Direction,
    Evidence,
    ExtractedEvent,
    ImpactAnalysis,
    ImpactCandidate,
    ImpactRecord,
    ImpactRule,
    ImpactType,
    TextChunk,
    TextSpan,
)
from hermes_bot.schemas import (
    Alias,
    EpistemicType,
)

MAX_TRAVERSAL_HOPS = 3


class ImpactExtractionError(ValueError):
    """Raised when input violates T08 boundaries or catalog constraints."""


class DeterministicImpactExtractor:
    """Extract events and impacts from supplied text using deterministic rules.

    Parameters
    ----------
    rules:
        Injectable rule table.  No rules are hard-coded by the extractor.
    catalog:
        Optional canonical catalog for alias resolution and entity lookup.
    graph:
        Optional typed temporal graph for relation traversal.
    max_hops:
        Hard cap on graph traversal; must be 0..3.
    """

    def __init__(
        self,
        rules: Sequence[ImpactRule] | None = None,
        catalog: Catalog | None = None,
        graph: EconomicGraph | None = None,
        max_hops: int = 2,
    ) -> None:
        self.rules = list(rules or [])
        self.catalog = catalog
        self.graph = graph
        if not 0 <= max_hops <= MAX_TRAVERSAL_HOPS:
            raise ImpactExtractionError(
                f"max_hops must be between 0 and {MAX_TRAVERSAL_HOPS}, got {max_hops}"
            )
        self.max_hops = max_hops

    def analyze(
        self,
        event: str,
        source: str = "news",
        entity_id: str = "",
        document_id: str = "",
        chunks: Sequence[TextChunk] | None = None,
    ) -> ImpactAnalysis:
        """Run deterministic extraction on ``event``.

        Returns an ``ImpactAnalysis`` in one of three states:

        * ``ok`` — at least one impact or candidate was produced.
        * ``degraded`` — extraction ran but no impacts were found.
        * ``error`` — required input/context is missing or invalid.
        """
        warnings: list[str] = []

        if not isinstance(event, str):
            return _error("event must be a string")
        if not isinstance(source, str):
            return _error("source must be a string")
        if not isinstance(entity_id, str):
            return _error("entity_id must be a string")

        # Reject model-shaped injection attempts in the source/model fields.
        if _looks_like_model_payload(source):
            return _error("source contains model-shaped payload; rejected")

        if not chunks:
            chunks = [TextChunk(chunk_id=_mk_chunk_id(document_id), text=event)]
        else:
            chunks = list(chunks)
            if not chunks:
                return _error("chunks must not be empty when provided")

        full_text = " ".join(c.text for c in chunks)
        if not full_text.strip() and not entity_id.strip():
            return _error("empty event and entity_id; nothing to analyze")

        matched_rules = _match_rules(self.rules, full_text)

        # Resolve seed entities from explicit entity_id and catalog aliases.
        seed_result = _resolve_seeds(
            catalog=self.catalog,
            text=full_text,
            entity_id=entity_id,
            chunks=chunks,
        )
        candidates = list(seed_result.candidates)
        warnings.extend(seed_result.warnings)
        seed_entities = seed_result.entities

        if not matched_rules and not seed_entities and not candidates:
            return ImpactAnalysis(
                event=_empty_event(event, source, entity_id, chunks, matched_rules),
                impacts=[],
                candidates=[],
                warnings=["no rules matched and no seed entity provided"],
                state="degraded",
                error=None,
            )

        # Gather evidence spans for every matched rule.
        rule_evidence: dict[str, Evidence] = {}
        for rule in matched_rules:
            spans = _find_spans(rule.trigger_phrase, chunks)
            evidence_ids: list[str] = []
            for span in spans:
                eid = f"span:{span.chunk_id}:{span.start}:{span.end}"
                evidence_ids.append(eid)
                rule_evidence[eid] = Evidence(evidence_id=eid, spans=(span,))
            if not spans:
                warnings.append(f"rule {rule.rule_id}: trigger matched but no span found")

        if not seed_entities:
            state: AnalysisState = "degraded"
            if matched_rules:
                warnings.append("rules matched but no seed entity resolved")
            event_obj = _build_event(
                event=event,
                source=source,
                entity_id=entity_id,
                chunks=chunks,
                matched_rules=[r.rule_id for r in matched_rules],
                rule_evidence=rule_evidence,
                seed_entities=[],
                epistemic_type=_combine_epistemic_types(matched_rules),
            )
            return ImpactAnalysis(
                event=event_obj,
                impacts=[],
                candidates=candidates,
                warnings=warnings,
                state=state,
                error=None,
            )

        if not matched_rules:
            event_obj = _build_event(
                event=event,
                source=source,
                entity_id=entity_id,
                chunks=chunks,
                matched_rules=[],
                rule_evidence={},
                seed_entities=list(seed_entities),
                epistemic_type=EpistemicType.UNKNOWN,
            )
            return ImpactAnalysis(
                event=event_obj,
                impacts=[],
                candidates=candidates,
                warnings=warnings + ["seed entities resolved but no rules matched"],
                state="degraded",
                error=None,
            )

        # Traverse graph to find impacted entities.
        impacts, traversal_warnings = _traverse_impacts(
            graph=self.graph,
            seed_entities=seed_entities,
            matched_rules=matched_rules,
            rule_evidence=rule_evidence,
            max_hops=self.max_hops,
            catalog=self.catalog,
        )
        warnings.extend(traversal_warnings)

        event_obj = _build_event(
            event=event,
            source=source,
            entity_id=entity_id,
            chunks=chunks,
            matched_rules=[r.rule_id for r in matched_rules],
            rule_evidence=rule_evidence,
            seed_entities=list(seed_entities),
            epistemic_type=_combine_epistemic_types(matched_rules),
        )

        state = "ok" if impacts else "degraded"
        return ImpactAnalysis(
            event=event_obj,
            impacts=impacts,
            candidates=candidates,
            warnings=warnings,
            state=state,
            error=None,
        )


def _mk_chunk_id(document_id: str) -> str:
    suffix = uuid.uuid4().hex[:16]
    if document_id:
        return f"chk_{document_id}_{suffix}"
    return f"chk_{suffix}"


def _looks_like_model_payload(value: str) -> bool:
    """Reject strings that look like LLM/model configuration injections."""
    lowered = value.lower()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "http://",
            "https://",
            "{",
            "[",
            "model=",
            "api_key=",
        )
    )


def _error(message: str) -> ImpactAnalysis:
    return ImpactAnalysis(
        event=ExtractedEvent(
            event_id=f"evt_{uuid.uuid4().hex}",
            event_type="error",
            normalized_claim_text="",
            stance_direction=None,
            epistemic_type=EpistemicType.UNKNOWN,
            evidence=[],
            matched_rules=[],
            source_document_ids=[],
            source_chunk_ids=[],
            affected_entity_ids=[],
            uncertainty_fields={"error": message},
            extracted_at=datetime.now(UTC),
        ),
        impacts=[],
        candidates=[],
        warnings=[],
        state="error",
        error=message,
    )


def _empty_event(
    event: str,
    source: str,
    entity_id: str,
    chunks: list[TextChunk],
    matched_rules: list[ImpactRule],
) -> ExtractedEvent:
    return _build_event(
        event=event,
        source=source,
        entity_id=entity_id,
        chunks=chunks,
        matched_rules=[r.rule_id for r in matched_rules],
        rule_evidence={},
        seed_entities=[],
        epistemic_type=_combine_epistemic_types(matched_rules),
    )


def _build_event(
    event: str,
    source: str,
    entity_id: str,
    chunks: list[TextChunk],
    matched_rules: list[str],
    rule_evidence: dict[str, Evidence],
    seed_entities: list[str],
    epistemic_type: EpistemicType,
) -> ExtractedEvent:
    evidence = list(rule_evidence.values())
    return ExtractedEvent(
        event_id=f"evt_{uuid.uuid4().hex}",
        event_type=source,
        normalized_claim_text=event.strip(),
        stance_direction=None,
        epistemic_type=epistemic_type,
        evidence=evidence,
        matched_rules=matched_rules,
        source_document_ids=[],
        source_chunk_ids=[c.chunk_id for c in chunks],
        affected_entity_ids=seed_entities,
        uncertainty_fields={},
        extracted_at=datetime.now(UTC),
    )


def _match_rules(rules: list[ImpactRule], text: str) -> list[ImpactRule]:
    lowered = text.lower()
    matched: list[ImpactRule] = []
    for rule in rules:
        trigger = rule.trigger_phrase.lower()
        if trigger not in lowered:
            continue
        if any(req.lower() not in lowered for req in rule.required_phrases):
            continue
        if any(forb.lower() in lowered for forb in rule.forbidden_phrases):
            continue
        matched.append(rule)
    return matched


def _find_spans(phrase: str, chunks: list[TextChunk]) -> list[TextSpan]:
    spans: list[TextSpan] = []
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    for chunk in chunks:
        for match in pattern.finditer(chunk.text):
            spans.append(
                TextSpan(
                    chunk_id=chunk.chunk_id,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(0),
                )
            )
    return spans


@dataclass
class _SeedResult:
    entities: set[str]
    candidates: list[ImpactCandidate]
    warnings: list[str]


def _resolve_seeds(
    catalog: Catalog | None,
    text: str,
    entity_id: str,
    chunks: list[TextChunk],
) -> _SeedResult:
    entities: set[str] = set()
    candidates: list[ImpactCandidate] = []
    warnings: list[str] = []

    if catalog is None:
        if entity_id.strip():
            warnings.append("catalog unavailable; entity_id cannot be validated")
        return _SeedResult(entities=entities, candidates=candidates, warnings=warnings)

    # Validate explicit entity_id if provided.
    if entity_id.strip():
        entity = catalog.entity.get(entity_id)
        if entity is not None:
            entities.add(entity_id)
        else:
            warnings.append(f"explicit entity_id {entity_id!r} not found in catalog")

    # Find all catalog aliases present in the text.
    aliases = _load_aliases(catalog)
    alias_to_entities: dict[str, dict[str, Alias]] = defaultdict(dict)
    for alias in aliases:
        if alias.resolution_status.value == "unresolved":
            continue
        if _alias_in_text(alias.alias, text):
            alias_to_entities[alias.alias][alias.entity_id] = alias

    for alias, entity_map in alias_to_entities.items():
        if len(entity_map) > 1:
            for alias_rec in entity_map.values():
                ent = catalog.entity.get(alias_rec.entity_id)
                name = ent.canonical_name if ent else alias_rec.entity_id
                entity_type = ent.entity_type.value if ent else "unknown"
                candidates.append(
                    ImpactCandidate(
                        entity_id=alias_rec.entity_id,
                        canonical_name=name,
                        entity_type=entity_type,
                        alias=alias,
                        reason_code="ambiguous_alias",
                    )
                )
            warnings.append(
                f"ambiguous alias {alias!r} resolves to "
                f"{[c.entity_id for c in candidates if c.alias == alias]}"
            )
        else:
            entities.update(entity_map.keys())

    return _SeedResult(entities=entities, candidates=candidates, warnings=warnings)


def _load_aliases(catalog: Catalog) -> list[Alias]:
    conn = catalog.conn
    rows = conn.execute(
        "SELECT alias, namespace, entity_id, valid_from, valid_to, "
        "resolution_status, created_at FROM alias"
    ).fetchall()
    return [
        Alias(
            alias=row["alias"],
            namespace=row["namespace"],
            entity_id=row["entity_id"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            resolution_status=_parse_resolution(row["resolution_status"]),
            created_at=_parse_dt_required(row["created_at"]),
        )
        for row in rows
    ]


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _parse_dt_required(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _parse_resolution(value: str):
    from hermes_bot.schemas import ResolutionStatus

    return ResolutionStatus(value)


def _alias_in_text(alias: str, text: str) -> bool:
    # Exact case-insensitive substring match, but require word boundaries for
    # short aliases (<=4 chars) to avoid false positives.
    lowered_alias = alias.lower()
    lowered_text = text.lower()
    if len(lowered_alias) <= 4:
        pattern = re.compile(r"\b" + re.escape(lowered_alias) + r"\b")
        return bool(pattern.search(lowered_text))
    return lowered_alias in lowered_text


def _traverse_impacts(
    graph: EconomicGraph | None,
    seed_entities: set[str],
    matched_rules: list[ImpactRule],
    rule_evidence: dict[str, Evidence],
    max_hops: int,
    catalog: Catalog | None,
) -> tuple[list[ImpactRecord], list[str]]:
    impacts: list[ImpactRecord] = []
    warnings: list[str] = []

    if graph is None or catalog is None:
        warnings.append("graph or catalog unavailable; cannot traverse relations")
        return impacts, warnings

    now = datetime.now(UTC)
    # Map target entity -> list of (rule_id, path) that reached it.
    reachable: dict[str, list[tuple[str, GraphPath]]] = defaultdict(list)
    for seed in seed_entities:
        for rule in matched_rules:
            allowed = rule.allowed_relation_types
            try:
                paths = graph.traverse(
                    seed,
                    as_of=now,
                    max_hops=max_hops,
                    direction="both",
                    allowed_types=allowed,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"traversal failed for {seed}: {exc}")
                continue
            for path in paths:
                target = path.path[-1]
                reachable[target].append((rule.rule_id, path))

    # Build per-entity rule sets and direction sets to detect conflict.
    rule_by_id = {r.rule_id: r for r in matched_rules}
    entity_rules: dict[str, set[str]] = defaultdict(set)
    entity_paths: dict[str, list[GraphPath]] = defaultdict(list)
    entity_directions: dict[str, set[Direction]] = defaultdict(set)
    for target, rule_paths in reachable.items():
        for rule_id, path in rule_paths:
            entity_rules[target].add(rule_id)
            entity_paths[target].append(path)
            entity_directions[target].add(rule_by_id[rule_id].direction)

    for target, directions in entity_directions.items():
        direction = _merge_directions(directions)
        if len(directions) > 1:
            warnings.append(
                f"conflicting directions for {target}: "
                f"{sorted(directions)}; evidence ids recorded"
            )

        # Gather evidence IDs: rule spans + graph edge evidence.
        evidence_ids: list[str] = []
        for eid in rule_evidence:
            evidence_ids.append(eid)
        for path in entity_paths[target]:
            for hop in path.hops:
                for eid in hop.evidence_ids:
                    evidence_ids.append(eid)

        # Determine impact type and confidence from rules that reached target.
        reaching_rules = [rule_by_id[rid] for rid in entity_rules[target]]
        impact_type = _merge_impact_types(reaching_rules)
        confidence_label, subjective_confidence = _merge_confidence(reaching_rules)

        record = ImpactRecord(
            entity_id=target,
            impact_type=impact_type,
            direction=direction,
            horizon_days=list(reaching_rules[0].horizon_days) if reaching_rules else [1, 5, 20],
            estimated_return=None,
            estimated_return_semantics="none",
            subjective_confidence=subjective_confidence,
            confidence_label=confidence_label,
            calibration_id=None,
            evidence_ids=sorted(set(evidence_ids)),
            reason_codes=sorted(entity_rules[target]),
            warnings=[],
        )
        impacts.append(record)

    return impacts, warnings


def _merge_directions(directions: set[Direction]) -> Direction:
    if len(directions) == 1:
        return next(iter(directions))
    if directions == {"positive", "negative"}:
        return "mixed"
    if "mixed" in directions:
        return "mixed"
    if "unknown" in directions and len(directions) == 2:
        other = (directions - {"unknown"}).pop()
        return other
    return "unknown"


def _merge_impact_types(rules: list[ImpactRule]) -> ImpactType:
    if not rules:
        return "other"
    # Prefer demand/supply/policy/rates over other.
    priority = ["demand", "supply", "policy", "rates", "revenue", "cost", "other"]
    types = {r.impact_type for r in rules}
    for t in priority:
        if t in types:
            return t  # type: ignore[return-value]
    return "other"


def _merge_confidence(rules: list[ImpactRule]) -> tuple[str, float]:
    label_to_value = {"low": 0.25, "medium": 0.5, "high": 0.75}
    if not rules:
        return "low", 0.0
    labels = [r.confidence_label for r in rules]
    # Highest confidence wins.
    best = max(labels, key=lambda x: label_to_value.get(x, 0.0))
    return best, label_to_value.get(best, 0.5)


def _combine_epistemic_types(rules: list[ImpactRule]) -> EpistemicType:
    if not rules:
        return EpistemicType.UNKNOWN
    types = {r.epistemic_type for r in rules}
    if len(types) == 1:
        return next(iter(types))
    # Preserve allegation if any rule marks it.
    if EpistemicType.ALLEGATION in types:
        return EpistemicType.ALLEGATION
    if EpistemicType.FORECAST in types:
        return EpistemicType.FORECAST
    return EpistemicType.UNKNOWN
