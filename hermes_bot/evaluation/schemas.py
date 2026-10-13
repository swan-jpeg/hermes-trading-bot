"""Pydantic schemas for the T11 evaluation harness."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationConfig(BaseModel):
    """Reproducible evaluator configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split_time: datetime | None = None
    train_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = 42
    cost_rate: float = Field(default=0.001, ge=0.0)
    default_horizon: str = "1d"
    version: str = "t11/v1"


class HorizonOutcomes(BaseModel):
    """Future outcomes per horizon (e.g. 1d, 5d, 20d)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_days: dict[str, float] = Field(default_factory=dict)


class EvaluationRecord(BaseModel):
    """One prediction-decision record with strict as-of timestamps.

    ``event_available_at`` is when the source information became usable.
    ``decision_at`` is when the prediction was (or would be) made.
    ``label_available_at`` is when the ground-truth label became known.
    Strict ordering requires ``event_available_at <= decision_at < label_available_at``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    version: str = "t11/v1"
    event_available_at: datetime
    label_available_at: datetime
    decision_at: datetime
    predicted_entities: list[str] = Field(default_factory=list)
    predicted_probabilities: dict[str, float] = Field(default_factory=dict)
    target_weights: dict[str, float] = Field(default_factory=dict)
    true_entities: list[str] = Field(default_factory=list)
    outcomes: dict[str, HorizonOutcomes] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_as_of(self) -> EvaluationRecord:
        if self.event_available_at > self.decision_at:
            raise ValueError(
                f"record {self.record_id}: event_available_at ({self.event_available_at}) "
                f"must be <= decision_at ({self.decision_at})"
            )
        if self.label_available_at <= self.decision_at:
            raise ValueError(
                f"record {self.record_id}: label_available_at ({self.label_available_at}) "
                f"must be > decision_at ({self.decision_at})"
            )
        return self

    @model_validator(mode="after")
    def _check_probabilities(self) -> EvaluationRecord:
        for entity, prob in self.predicted_probabilities.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"record {self.record_id}: probability for {entity} must be in [0, 1]"
                )
        return self


class EvaluationReport(BaseModel):
    """Structured evaluation summary, always labeled synthetic/non-investable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    synthetic: bool = True
    investable: bool = False
    disclaimer: str = (
        "This is a synthetic, offline, non-investable evaluation. "
        "No result is validated alpha or profitability."
    )
    code_version: str
    config: dict[str, Any]
    dataset_hash: str
    seed: int
    n_train: int
    n_test: int
    n_entities: int
    split_time: datetime | None
    metrics: dict[str, float | None]
    baseline_metrics: dict[str, float | None]
