"""Deterministic, evidence-linked Impact Agent (T08).

Public entry points are the legacy-compatible ``ImpactAgent`` and
``LLMImpactAgent`` adapters, plus the new contracts and the underlying
``DeterministicImpactExtractor``.
"""
from __future__ import annotations

from hermes_bot.impact.contracts import (
    Evidence,
    ExtractedEvent,
    ImpactAnalysis,
    ImpactCandidate,
    ImpactRecord,
    ImpactRule,
    TextChunk,
    TextSpan,
)
from hermes_bot.impact.core import DeterministicImpactExtractor
from hermes_bot.impact.legacy import (
    ImpactAgent,
    ImpactResult,
    LLMImpactAgent,
    build_impact_agent,
)

__all__ = [
    "DeterministicImpactExtractor",
    "Evidence",
    "ExtractedEvent",
    "ImpactAgent",
    "ImpactAnalysis",
    "ImpactCandidate",
    "ImpactRecord",
    "ImpactResult",
    "ImpactRule",
    "LLMImpactAgent",
    "TextChunk",
    "TextSpan",
    "build_impact_agent",
]
