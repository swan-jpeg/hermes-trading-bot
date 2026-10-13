"""Tests for the T11 chronological impact/strategy evaluation harness."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hermes_bot.evaluation import (
    EvaluationConfig,
    EvaluationRecord,
    HorizonOutcomes,
    ImpactEvaluator,
    NoTradeBaseline,
    chronosplit,
    load_fixture,
)


def _dt(day: int, hour: int = 0) -> datetime:
    """Build a deterministic UTC datetime."""
    return datetime(2026, 1, day, hour, 0, 0, tzinfo=UTC)


def _record(
    rid: str,
    day: int,
    pred: list[str] | None = None,
    true: list[str] | None = None,
    probs: dict[str, float] | None = None,
    targets: dict[str, float] | None = None,
    outcomes: dict[str, dict[str, float]] | None = None,
    label_offset_hours: int = 12,
) -> EvaluationRecord:
    outcome_models = {
        e: HorizonOutcomes(horizon_days=v) for e, v in (outcomes or {}).items()
    }
    return EvaluationRecord(
        record_id=rid,
        event_available_at=_dt(day, 0),
        decision_at=_dt(day, 6),
        label_available_at=_dt(day, 6) + timedelta(hours=label_offset_hours),
        predicted_entities=pred or [],
        true_entities=true or [],
        predicted_probabilities=probs or {},
        target_weights=targets or {},
        outcomes=outcome_models,
    )


def test_leakage_guard_rejects_label_available_before_decision():
    """A label known at or before decision time is a leakage violation."""
    with pytest.raises(ValueError, match="label_available_at"):
        EvaluationRecord(
            record_id="rec_leak",
            event_available_at=_dt(1, 0),
            decision_at=_dt(1, 12),
            label_available_at=_dt(1, 12),
            predicted_entities=["a"],
            true_entities=["a"],
        )


def test_chronological_split_respects_decision_time():
    """Train contains earlier decisions; test contains later decisions."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"]),
        _record("r2", 2, pred=["b"], true=["b"]),
        _record("r3", 3, pred=["c"], true=["c"]),
        _record("r4", 4, pred=["d"], true=["d"]),
    ]
    config = EvaluationConfig(train_fraction=0.5)
    train, test, split_time = chronosplit(records, config)
    assert len(train) == 2
    assert len(test) == 2
    assert split_time is not None
    assert all(r.decision_at < split_time for r in train)
    assert all(r.decision_at >= split_time for r in test)


def test_later_known_labels_stay_in_test_split():
    """Labels created after the split boundary remain in the test split."""
    records = [
        _record("r1", 1, true=["a"]),
        _record("r2", 2, true=["a"]),
        _record("r3", 3, true=["b"]),
        _record("r4", 4, true=["b"]),
    ]
    config = EvaluationConfig(split_time=_dt(3, 0))
    train, test, _ = chronosplit(records, config)
    assert all(r.label_available_at > r.decision_at for r in test)
    assert len(train) == 2
    assert len(test) == 2


def test_repeated_seed_same_report():
    """Same fixture and seed produce identical dataset hash and metrics."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"]),
        _record("r2", 2, pred=["a"], true=["a"]),
        _record("r3", 3, pred=["b"], true=["b"]),
        _record("r4", 4, pred=["b"], true=["b"]),
    ]
    config = EvaluationConfig(seed=123)
    evaluator = ImpactEvaluator(config)
    report1 = evaluator.report(records)
    report2 = evaluator.report(records)
    assert report1.dataset_hash == report2.dataset_hash
    assert report1.metrics == report2.metrics


def test_baseline_is_no_trade():
    """No-trade baseline predicts nothing and keeps targets constant."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"], targets={"a": 0.5}),
        _record("r2", 2, pred=["b"], true=["b"], targets={"a": 0.3, "b": 0.2}),
        _record("r3", 3, pred=["c"], true=["c"], targets={"b": 0.4}),
        _record("r4", 4, pred=["d"], true=["d"], targets={"b": 0.4}),
    ]
    config = EvaluationConfig(train_fraction=0.5)
    evaluator = ImpactEvaluator(config)
    system, baseline = evaluator.evaluate(records)
    assert baseline.coverage == 0.0
    assert baseline.abstention_rate == 1.0
    assert baseline.target_turnover == 0.0


def test_empty_labels_metrics():
    """Empty true labels yield zero recall and coverage-driven precision."""
    records = [
        _record("r1", 1, pred=["a"], true=[]),
        _record("r2", 2, pred=["b"], true=[]),
    ]
    config = EvaluationConfig(split_time=_dt(2, 0))
    evaluator = ImpactEvaluator(config)
    system, _ = evaluator.evaluate(records)
    assert system.n_test == 1
    assert system.entity_recall == 0.0
    assert system.entity_precision == 0.0


def test_no_positives_empty_predictions():
    """No predictions for any test record yields zero precision and recall."""
    records = [
        _record("r1", 1, pred=[], true=["a"]),
        _record("r2", 2, pred=[], true=["b"]),
        _record("r3", 3, pred=[], true=["c"]),
    ]
    config = EvaluationConfig(split_time=_dt(2, 0))
    evaluator = ImpactEvaluator(config)
    system, _ = evaluator.evaluate(records)
    assert system.entity_precision is None or system.entity_precision == 0.0
    assert system.entity_recall == 0.0
    assert system.coverage == 0.0
    assert system.abstention_rate == 1.0


def test_entity_precision_recall_f1():
    """Hand-check P/R/F1 over two test records."""
    records = [
        _record("r1", 1, pred=["a", "b"], true=["a"]),
        _record("r2", 2, pred=["b", "c"], true=["b", "c", "d"]),
        _record("r3", 3, pred=["a"], true=["a"]),
        _record("r4", 4, pred=["c", "d"], true=["c"]),
    ]
    config = EvaluationConfig(split_time=_dt(3, 0))
    evaluator = ImpactEvaluator(config)
    system, _ = evaluator.evaluate(records)
    # Test records: r3 (TP=1, FP=0, FN=0), r4 (TP=1, FP=1, FN=0)
    # Total TP=2, FP=1, FN=0 -> P=2/3, R=1.0, F1=0.8
    assert system.entity_precision == pytest.approx(2 / 3)
    assert system.entity_recall == pytest.approx(1.0)
    assert system.entity_f1 == pytest.approx(0.8)


def test_brier_only_with_calibrated_probabilities():
    """Brier score is omitted when no probabilities are supplied."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"]),
        _record("r2", 2, pred=["a"], true=["a"]),
    ]
    evaluator = ImpactEvaluator(EvaluationConfig(split_time=_dt(2, 0)))
    system_no_probs, _ = evaluator.evaluate(records)
    assert system_no_probs.brier_score is None

    records_probs = [
        _record("r1", 1, pred=["a"], true=["a"], probs={"a": 0.9}),
        _record("r2", 2, pred=["a"], true=["a"], probs={"a": 0.1}),
    ]
    system_probs, _ = evaluator.evaluate(records_probs)
    # Only r2 is in the test split; Brier = (0.1 - 1)^2 = 0.81.
    assert system_probs.brier_score == pytest.approx(0.81)


def test_coverage_and_abstention():
    """Coverage and abstention are complementary fractions."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"]),
        _record("r2", 2, pred=[], true=["b"]),
        _record("r3", 3, pred=["c"], true=["c"]),
    ]
    evaluator = ImpactEvaluator(EvaluationConfig(split_time=_dt(2, 0)))
    system, _ = evaluator.evaluate(records)
    assert system.coverage == pytest.approx(0.5)
    assert system.abstention_rate == pytest.approx(0.5)
    assert system.coverage is not None
    assert system.abstention_rate is not None
    assert system.coverage + system.abstention_rate == pytest.approx(1.0)


def test_target_turnover_computed():
    """Target turnover is the average half L1 distance between consecutive targets."""
    records = [
        _record("r1", 1, targets={"a": 0.5, "b": 0.3}),
        _record("r2", 2, targets={"a": 0.4, "b": 0.3, "c": 0.2}),
        _record("r3", 3, targets={"a": 0.4, "b": 0.1, "c": 0.2}),
    ]
    evaluator = ImpactEvaluator(EvaluationConfig(split_time=_dt(2, 0)))
    system, _ = evaluator.evaluate(records)
    # r2 vs r1: |0.4-0.5|/2 + |0.3-0.3|/2 + |0.2-0|/2 = 0.05 + 0 + 0.1 = 0.15
    # r3 vs r2: |0.4-0.4|/2 + |0.1-0.3|/2 + |0.2-0.2|/2 = 0 + 0.1 + 0 = 0.1
    assert system.target_turnover == pytest.approx(0.125)


def test_returns_and_costs_when_outcomes_present():
    """Gross and net returns are computed only with explicit outcomes; costs subtracted."""
    records = [
        _record("r1", 1, targets={"a": 1.0}, outcomes={"a": {"1d": 0.05}}),
        _record("r2", 2, targets={"a": 1.0}, outcomes={"a": {"1d": 0.03}}),
        _record("r3", 3, targets={"a": 0.5}, outcomes={"a": {"1d": 0.04}}),
    ]
    config = EvaluationConfig(split_time=_dt(2, 0), cost_rate=0.01)
    evaluator = ImpactEvaluator(config)
    system, _ = evaluator.evaluate(records)
    assert system.gross_return is not None
    assert system.net_return is not None
    assert system.cost is not None
    # Test records: r2 (target unchanged -> gross 0.03, cost 0, net 0.03)
    # r3 (target 1.0 -> 0.5, turnover 0.25, cost 0.0025, gross 0.04*0.5=0.02, net 0.0175)
    assert system.gross_return == pytest.approx((0.03 + 0.02) / 2)
    assert system.cost == pytest.approx(0.0025 / 2)
    assert system.net_return == pytest.approx((0.03 + 0.0175) / 2)


def test_returns_omitted_without_outcomes():
    """Return metrics are omitted when the fixture has no horizon outcomes."""
    records = [
        _record("r1", 1, targets={"a": 1.0}),
        _record("r2", 2, targets={"a": 0.5}),
    ]
    evaluator = ImpactEvaluator(EvaluationConfig(split_time=_dt(2, 0)))
    system, _ = evaluator.evaluate(records)
    assert system.gross_return is None
    assert system.net_return is None
    assert system.cost is None


def test_report_files_written_to_tmp_path(tmp_path: Path):
    """JSON and Markdown reports are emitted to the supplied directory."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"]),
        _record("r2", 2, pred=["a"], true=["a"]),
    ]
    evaluator = ImpactEvaluator(EvaluationConfig(split_time=_dt(2, 0)))
    report = evaluator.report(records, output_dir=tmp_path)
    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "evaluation_report.md").exists()
    loaded = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert loaded["synthetic"] is True
    assert loaded["investable"] is False
    assert "non-investable" in loaded["disclaimer"].lower()
    assert report.dataset_hash == loaded["dataset_hash"]


def test_jsonl_fixture_round_trip(tmp_path: Path):
    """EvaluationRecords can be written to and loaded from a JSONL fixture file."""
    fixture_path = tmp_path / "fixture.jsonl"
    records = [
        _record("r1", 1, pred=["a"], true=["a"], probs={"a": 0.7}),
        _record("r2", 2, pred=["b"], true=["b"], outcomes={"b": {"1d": 0.02}}),
    ]
    with fixture_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    loaded = load_fixture(fixture_path)
    assert len(loaded) == 2
    assert loaded[0].record_id == "r1"
    assert loaded[0].predicted_probabilities == {"a": 0.7}
    assert loaded[1].outcomes["b"].horizon_days == {"1d": 0.02}


def test_no_trade_baseline_carries_forward_targets():
    """Baseline holds the last training target map constant during the test split."""
    baseline = NoTradeBaseline()
    rec1 = _record("r1", 1, targets={"a": 0.5})
    rec2 = _record("r2", 2, targets={"a": 0.2})
    rec3 = _record("r3", 3, targets={"a": 0.9})
    baseline.fit([rec1, rec2])
    pred3, targets3, _ = baseline.predict(rec3)
    assert pred3 == []
    assert targets3 == {"a": 0.2}
    baseline.update(rec3)
    pred4, targets4, _ = baseline.predict(rec3)
    assert targets4 == {"a": 0.2}


def test_dataset_hash_method_is_sha256():
    """Dataset hash is a 64-character lowercase hex SHA-256 string."""
    records = [
        _record("r1", 1, pred=["a"], true=["a"]),
        _record("r2", 2, pred=["b"], true=["b"]),
    ]
    evaluator = ImpactEvaluator()
    report = evaluator.report(records)
    assert len(report.dataset_hash) == 64
    assert all(c in "0123456789abcdef" for c in report.dataset_hash)


def test_config_seed_recorded():
    """Report records seed, config and code version for reproducibility."""
    records = [_record("r1", 1, pred=["a"], true=["a"])]
    config = EvaluationConfig(seed=987, default_horizon="5d")
    evaluator = ImpactEvaluator(config)
    report = evaluator.report(records)
    assert report.seed == 987
    assert report.config["seed"] == 987
    assert report.config["default_horizon"] == "5d"
    assert report.config["dataset_hash_method"] == "sha256 of canonical sorted fixture JSON"
    assert report.code_version != "unknown"
