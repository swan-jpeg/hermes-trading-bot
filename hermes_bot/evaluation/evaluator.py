"""Chronological impact/strategy evaluation with leakage guards."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_bot.evaluation.schemas import (
    EvaluationConfig,
    EvaluationRecord,
    EvaluationReport,
)


def _canonical_json(value: Any) -> str:
    """Stable JSON serialization for hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def load_fixture(path: Path) -> list[EvaluationRecord]:
    """Load a versioned JSONL fixture into ``EvaluationRecord`` objects."""
    records: list[EvaluationRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(EvaluationRecord.model_validate_json(line))
    return records


def _dataset_hash(records: list[EvaluationRecord]) -> str:
    """SHA-256 of the canonical, sorted fixture contents."""
    payload = [json.loads(r.model_dump_json()) for r in records]
    payload = sorted(payload, key=lambda d: d["record_id"])
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _code_version() -> str:
    """Read version from pyproject.toml; fallback to unknown."""
    try:
        import tomllib

        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with path.open("rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version", "unknown")
    except Exception:  # pragma: no cover - defensive fallback
        return "unknown"


def chronosplit(
    records: list[EvaluationRecord],
    config: EvaluationConfig,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], datetime | None]:
    """Split records chronologically by ``decision_at``.

    If ``config.split_time`` is provided it is used directly.  Otherwise the split
    is derived from ``train_fraction`` using sorted decision times.  The first
    ``idx`` records (by time) become the training split.
    """
    sorted_records = sorted(records, key=lambda r: r.decision_at)
    if config.split_time is not None:
        split_time = config.split_time
        train = [r for r in sorted_records if r.decision_at < split_time]
        test = [r for r in sorted_records if r.decision_at >= split_time]
        return train, test, split_time

    n = len(sorted_records)
    if n == 0:
        return [], [], None
    idx = max(1, int(round(n * config.train_fraction)))
    idx = min(idx, n - 1) if n > 1 else n
    train = sorted_records[:idx]
    test = sorted_records[idx:]
    split_time = test[0].decision_at if test else train[-1].decision_at
    return train, test, split_time


@dataclass
class EvaluationResult:
    """Raw metric bundle produced by ``ImpactEvaluator.evaluate``."""

    n_train: int = 0
    n_test: int = 0
    n_entities: int = 0
    entity_precision: float | None = None
    entity_recall: float | None = None
    entity_f1: float | None = None
    coverage: float | None = None
    abstention_rate: float | None = None
    target_turnover: float | None = None
    brier_score: float | None = None
    gross_return: float | None = None
    net_return: float | None = None
    cost: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "n_train": float(self.n_train),
            "n_test": float(self.n_test),
            "n_entities": float(self.n_entities),
            "entity_precision": self.entity_precision,
            "entity_recall": self.entity_recall,
            "entity_f1": self.entity_f1,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "target_turnover": self.target_turnover,
            "brier_score": self.brier_score,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "cost": self.cost,
        }


class NoTradeBaseline:
    """Naive baseline: predict no entities and keep target weights constant."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self.previous_targets: dict[str, float] = {}

    def fit(self, train: list[EvaluationRecord]) -> None:
        """No-trade baseline holds the last training target map constant."""
        if train:
            last = max(train, key=lambda r: r.decision_at)
            self.previous_targets = dict(last.target_weights)
        else:
            self.previous_targets = {}

    def predict(
        self,
        record: EvaluationRecord,
    ) -> tuple[list[str], dict[str, float], dict[str, float] | None]:
        """Return empty predictions and unchanged target weights."""
        return [], dict(self.previous_targets), None

    def update(self, record: EvaluationRecord) -> None:
        """No-trade baseline does not change its target map."""
        return


class ImpactEvaluator:
    """Offline evaluator with strict as-of and reproducibility guarantees."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self.baseline = NoTradeBaseline(self.config)

    def _entity_metrics(
        self,
        test: list[EvaluationRecord],
    ) -> tuple[float | None, float | None, float | None, int, int, int]:
        """Aggregate precision/recall/F1 over all test records."""
        tp = fp = fn = 0
        for rec in test:
            pred = set(rec.predicted_entities)
            true = set(rec.true_entities)
            tp += len(pred & true)
            fp += len(pred - true)
            fn += len(true - pred)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1: float | None = None
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1, tp, fp, fn

    def _coverage_abstention(
        self,
        test: list[EvaluationRecord],
    ) -> tuple[float | None, float | None]:
        if not test:
            return None, None
        n_pred = sum(1 for rec in test if rec.predicted_entities)
        coverage = n_pred / len(test)
        abstention = (len(test) - n_pred) / len(test)
        return coverage, abstention

    def _target_turnover(
        self,
        train: list[EvaluationRecord],
        test: list[EvaluationRecord],
    ) -> float | None:
        if not test:
            return None
        sorted_test = sorted(test, key=lambda r: r.decision_at)
        prev: dict[str, float] = {}
        if train:
            last_train = max(train, key=lambda r: r.decision_at)
            prev = dict(last_train.target_weights)
        turnovers: list[float] = []
        for rec in sorted_test:
            cur = rec.target_weights
            keys = set(prev) | set(cur)
            turnover = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys) / 2.0
            turnovers.append(turnover)
            prev = dict(cur)
        return sum(turnovers) / len(turnovers) if turnovers else None

    def _brier_score(
        self,
        test: list[EvaluationRecord],
    ) -> float | None:
        """Brier score only when calibrated probabilities are supplied."""
        scores: list[float] = []
        has_probs = False
        for rec in test:
            if not rec.predicted_probabilities:
                continue
            has_probs = True
            true = set(rec.true_entities)
            universe = set(rec.predicted_probabilities) | true
            for entity in universe:
                p = rec.predicted_probabilities.get(entity, 0.0)
                y = 1.0 if entity in true else 0.0
                scores.append((p - y) ** 2)
        if not has_probs:
            return None
        return sum(scores) / len(scores) if scores else 0.0

    def _returns(
        self,
        train: list[EvaluationRecord],
        test: list[EvaluationRecord],
    ) -> tuple[float | None, float | None, float | None]:
        """Gross/net returns only when future outcomes are explicitly supplied."""
        if not test:
            return None, None, None
        sorted_test = sorted(test, key=lambda r: r.decision_at)
        prev: dict[str, float] = {}
        if train:
            last_train = max(train, key=lambda r: r.decision_at)
            prev = dict(last_train.target_weights)

        grosses: list[float] = []
        nets: list[float] = []
        costs: list[float] = []
        has_outcomes = False
        for rec in sorted_test:
            if not rec.outcomes:
                prev = dict(rec.target_weights)
                continue
            has_outcomes = True
            gross = 0.0
            for entity, weight in rec.target_weights.items():
                outcome = rec.outcomes.get(entity)
                if outcome is None:
                    continue
                value = outcome.horizon_days.get(self.config.default_horizon)
                if value is None:
                    continue
                gross += weight * value

            # Cost = turnover from previous target to current target, scaled by cost_rate.
            cur = rec.target_weights
            keys = set(prev) | set(cur)
            turnover = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys) / 2.0
            cost = turnover * self.config.cost_rate
            grosses.append(gross)
            nets.append(gross - cost)
            costs.append(cost)
            prev = dict(cur)

        if not has_outcomes:
            return None, None, None
        gross = sum(grosses) / len(grosses) if grosses else 0.0
        net = sum(nets) / len(nets) if nets else 0.0
        cost = sum(costs) / len(costs) if costs else 0.0
        return gross, net, cost

    def evaluate(
        self,
        records: list[EvaluationRecord],
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Return (system_result, baseline_result) for the supplied fixture."""
        train, test, split_time = chronosplit(records, self.config)
        n_entities = len(
            {
                e
                for rec in records
                for e in (rec.predicted_entities + rec.true_entities + list(rec.target_weights))
            }
        )

        system = self._evaluate_split(train, test, n_entities)
        baseline = self._evaluate_baseline(train, test, n_entities)
        self.split_time = split_time
        return system, baseline

    def _evaluate_split(
        self,
        train: list[EvaluationRecord],
        test: list[EvaluationRecord],
        n_entities: int,
    ) -> EvaluationResult:
        precision, recall, f1, _, _, _ = self._entity_metrics(test)
        coverage, abstention = self._coverage_abstention(test)
        turnover = self._target_turnover(train, test)
        brier = self._brier_score(test)
        gross, net, cost = self._returns(train, test)
        return EvaluationResult(
            n_train=len(train),
            n_test=len(test),
            n_entities=n_entities,
            entity_precision=precision,
            entity_recall=recall,
            entity_f1=f1,
            coverage=coverage,
            abstention_rate=abstention,
            target_turnover=turnover,
            brier_score=brier,
            gross_return=gross,
            net_return=net,
            cost=cost,
        )

    def _evaluate_baseline(
        self,
        train: list[EvaluationRecord],
        test: list[EvaluationRecord],
        n_entities: int,
    ) -> EvaluationResult:
        self.baseline.fit(train)
        baseline_records: list[EvaluationRecord] = []
        for rec in sorted(test, key=lambda r: r.decision_at):
            pred, targets, _ = self.baseline.predict(rec)
            baseline_records.append(
                EvaluationRecord(
                    record_id=rec.record_id,
                    decision_at=rec.decision_at,
                    event_available_at=rec.event_available_at,
                    label_available_at=rec.label_available_at,
                    predicted_entities=pred,
                    true_entities=rec.true_entities,
                    target_weights=targets,
                    outcomes=rec.outcomes,
                )
            )
            self.baseline.update(rec)

        precision, recall, f1, _, _, _ = self._entity_metrics(baseline_records)
        coverage, abstention = self._coverage_abstention(baseline_records)
        turnover = self._target_turnover(train, baseline_records)
        brier = self._brier_score(baseline_records)
        gross, net, cost = self._returns(train, baseline_records)
        return EvaluationResult(
            n_train=len(train),
            n_test=len(test),
            n_entities=n_entities,
            entity_precision=precision,
            entity_recall=recall,
            entity_f1=f1,
            coverage=coverage,
            abstention_rate=abstention,
            target_turnover=turnover,
            brier_score=brier,
            gross_return=gross,
            net_return=net,
            cost=cost,
        )

    def report(
        self,
        records: list[EvaluationRecord],
        output_dir: Path | None = None,
    ) -> EvaluationReport:
        """Run evaluation and emit a reproducible JSON/Markdown report."""
        system, baseline = self.evaluate(records)
        dataset_hash = _dataset_hash(records)
        code_version = _code_version()
        config_dict = json.loads(self.config.model_dump_json())
        config_dict["code_version"] = code_version
        config_dict["dataset_hash_method"] = "sha256 of canonical sorted fixture JSON"

        report = EvaluationReport(
            synthetic=True,
            investable=False,
            code_version=code_version,
            config=config_dict,
            dataset_hash=dataset_hash,
            seed=self.config.seed,
            n_train=system.n_train,
            n_test=system.n_test,
            n_entities=system.n_entities,
            split_time=getattr(self, "split_time", None),
            metrics=system.to_dict(),
            baseline_metrics=baseline.to_dict(),
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / "evaluation_report.json"
            md_path = output_dir / "evaluation_report.md"
            json_path.write_text(report.model_dump_json(indent=2))
            md_path.write_text(self._render_markdown(report))

        return report

    def _render_markdown(self, report: EvaluationReport) -> str:
        lines = [
            "# T11 Evaluation Report",
            "",
            f"**Synthetic / non-investable**: {report.synthetic}",
            "",
            f"> {report.disclaimer}",
            "",
            "## Reproducibility",
            "",
            f"- **Code version**: `{report.code_version}`",
            f"- **Dataset hash**: `{report.dataset_hash}`",
            f"- **Seed**: `{report.seed}`",
            f"- **Config**: `{_canonical_json(report.config)}`",
            "",
            "## Split",
            "",
            f"- **Train records**: {report.n_train}",
            f"- **Test records**: {report.n_test}",
            f"- **Entity universe**: {report.n_entities}",
            f"- **Split time**: {report.split_time}",
            "",
            "## System metrics",
            "",
            "| metric | value |",
            "|--------|-------|",
        ]
        for key, value in report.metrics.items():
            if value is None:
                lines.append(f"| {key} | omitted |")
            else:
                lines.append(f"| {key} | {value:.6f} |")
        lines.extend([
            "",
            "## Naive / no-trade baseline metrics",
            "",
            "| metric | value |",
            "|--------|-------|",
        ])
        for key, value in report.baseline_metrics.items():
            if value is None:
                lines.append(f"| {key} | omitted |")
            else:
                lines.append(f"| {key} | {value:.6f} |")
        lines.extend([
            "",
            "## Limitations",
            "",
            "- All inputs are synthetic fixtures.",
            "- Brier score is reported only when calibrated probabilities are supplied.",
            "- Returns are reported only when future outcomes are explicitly present.",
            "- No parameter selection was performed on the test split.",
            "- This report does not constitute investment advice or validated alpha.",
            "",
        ])
        return "\n".join(lines)
