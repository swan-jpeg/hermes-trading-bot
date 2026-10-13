"""Point-in-time impact and strategy evaluation harness (T11).

Offline evaluator for event-to-entity retrieval/classification and strategy
target proposals.  Strict as-of filtering, synthetic fixtures only, no
profitability claims.
"""
from __future__ import annotations

from hermes_bot.evaluation.evaluator import (
    EvaluationResult,
    ImpactEvaluator,
    NoTradeBaseline,
    chronosplit,
    load_fixture,
)
from hermes_bot.evaluation.schemas import (
    EvaluationConfig,
    EvaluationRecord,
    EvaluationReport,
    HorizonOutcomes,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationRecord",
    "EvaluationReport",
    "EvaluationResult",
    "HorizonOutcomes",
    "ImpactEvaluator",
    "NoTradeBaseline",
    "chronosplit",
    "load_fixture",
]
