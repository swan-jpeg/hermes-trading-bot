"""Canonical contracts for deterministic, evidence-linked impact extraction (T08).

These dataclasses replace the old heuristic ``ImpactResult`` with a traceable
structure: every impact points to source text spans and/or catalog relation
edges, and ``estimated_return`` is always null.  Confidence is subjective and
explicitly labeled (never "probability").
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from hermes_bot.schemas import EpistemicType, RelationType

Direction = Literal["positive", "negative", "mixed", "unknown"]
ImpactType = Literal["revenue", "cost", "supply", "demand", "policy", "rates", "other"]
AnalysisState = Literal["ok", "degraded", "error"]


@dataclass(frozen=True)
class ImpactRule:
    """Injectable deterministic rule for event/impact extraction.

    A rule matches when its ``trigger_phrase`` occurs in the text, every
    ``required_phrases`` is present, and none of ``forbidden_phrases`` is
    present.  Matching is case-insensitive exact-substring only: no NLP
    inference is performed.
    """

    rule_id: str
    trigger_phrase: str
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    allowed_relation_types: tuple[RelationType, ...] | None = None
    direction: Direction = "unknown"
    impact_type: ImpactType = "other"
    horizon_days: tuple[int, ...] = (1, 5, 20)
    confidence_label: str = "medium"
    epistemic_type: EpistemicType = EpistemicType.UNKNOWN

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must not be empty")
        if not self.trigger_phrase:
            raise ValueError("trigger_phrase must not be empty")
        if self.direction not in ("positive", "negative", "mixed", "unknown"):
            raise ValueError(f"invalid direction {self.direction!r}")
        if not isinstance(self.epistemic_type, EpistemicType):
            self.__dict__["epistemic_type"] = EpistemicType(self.epistemic_type)


@dataclass(frozen=True)
class TextChunk:
    """A chunk of source text with a stable identifier."""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class TextSpan:
    """Exact character span inside a chunk."""

    chunk_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("span start must be non-negative")
        if self.end < self.start:
            raise ValueError("span end must be >= start")


@dataclass(frozen=True)
class Evidence:
    """A piece of evidence: either a text span or a catalog relation edge."""

    evidence_id: str
    spans: tuple[TextSpan, ...] = ()


@dataclass
class ExtractedEvent:
    """Normalized event/claim extracted from supplied text."""

    event_id: str
    event_type: str
    normalized_claim_text: str
    stance_direction: Direction | None
    epistemic_type: EpistemicType
    evidence: list[Evidence]
    matched_rules: list[str]
    source_document_ids: list[str]
    source_chunk_ids: list[str]
    affected_entity_ids: list[str]
    uncertainty_fields: dict[str, object]
    extracted_at: datetime


@dataclass
class ImpactCandidate:
    """One candidate entity for an ambiguous alias."""

    entity_id: str
    canonical_name: str
    entity_type: str
    alias: str
    reason_code: str


@dataclass
class ImpactRecord:
    """Traceable impact on a single canonical entity or instrument."""

    entity_id: str
    impact_type: ImpactType
    direction: Direction
    horizon_days: list[int]
    estimated_return: None
    estimated_return_semantics: Literal["none", "association", "causal"]
    subjective_confidence: float
    confidence_label: str
    calibration_id: str | None
    evidence_ids: list[str]
    reason_codes: list[str]
    warnings: list[str]


@dataclass
class ImpactAnalysis:
    """Full deterministic impact analysis for one input."""

    event: ExtractedEvent
    impacts: list[ImpactRecord]
    candidates: list[ImpactCandidate]
    warnings: list[str]
    state: AnalysisState
    error: str | None
