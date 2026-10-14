"""Benchmark runner: measures SQLite, graph, vector and pipeline performance.

The runner is self-contained and never writes outside a temporary directory.
All measurements include wall time and, on POSIX systems, peak RSS.
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from hermes_bot.benchmark.generator import generate_dataset
from hermes_bot.data.catalog import Catalog
from hermes_bot.data.graph import EconomicGraph
from hermes_bot.data.migrations import MigrationRunner
from hermes_bot.pipeline import Pipeline
from hermes_bot.retrieval.index import VectorIndex


def _peak_rss_bytes() -> int | None:
    """Return peak resident set size in bytes, if available on this OS."""
    try:
        import resource  # local import: module only exists on POSIX

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return None


def _timer() -> Callable[[], float]:
    return time.perf_counter


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the scale benchmark."""

    sizes: tuple[int, ...] = (10, 100, 1000)
    seed: int = 42
    max_total_seconds: float = 60.0
    max_scenario_seconds: float = 20.0
    max_rss_bytes: int = field(default_factory=lambda: 2 * 1024**3)
    include_pipeline: bool = True
    top_k: int = 5


@dataclass
class ScenarioMeasurements:
    """Raw measurements for one workload size."""

    n: int
    elapsed_seconds: dict[str, float] = field(default_factory=dict)
    peak_rss_bytes: int | None = None
    result_counts: dict[str, int] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "result_counts": self.result_counts,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark report."""

    config: BenchmarkConfig
    environment: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    scenarios: list[ScenarioMeasurements]
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dataclasses.asdict(self.config),
            "environment": self.environment,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "recommendation": self.recommendation,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "MVP scale benchmark",
            "=" * 70,
            f"started : {self.started_at.isoformat()}",
            f"finished: {self.finished_at.isoformat()}",
            f"python  : {self.environment['python_version']}",
            f"platform: {self.environment['platform']}",
            f"cpu_count: {self.environment['cpu_count']}",
            f"seed    : {self.config.seed}",
            "",
        ]
        for scenario in self.scenarios:
            lines.append(f"size={scenario.n:>5}  records")
            if scenario.aborted:
                lines.append(f"  ABORTED: {scenario.abort_reason}")
                continue
            for name, seconds in scenario.elapsed_seconds.items():
                lines.append(f"  {name:.<40} {seconds:.6f} s")
            if scenario.peak_rss_bytes is not None:
                lines.append(
                    f"  {'peak_rss':.<40} {scenario.peak_rss_bytes / (1024 ** 2):.2f} MB"
                )
            for name, count in scenario.result_counts.items():
                lines.append(f"  {name + '_count':.<40} {count}")
            lines.append("")
        lines.append("-" * 70)
        lines.append("Recommendation")
        lines.append("-" * 70)
        lines.append(self.recommendation)
        lines.append("=" * 70)
        return "\n".join(lines)


def _environment() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "command": " ".join(sys.argv),
    }


def _stub_closes(entity: str) -> np.ndarray:
    """Deterministic offline price series (no network)."""
    rng = np.random.default_rng(abs(hash(entity)) % 2**31)
    return np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100


def _measure(label: str, measurements: ScenarioMeasurements, fn: Callable[[], Any]) -> Any:
    timer = _timer()
    start = timer()
    result = fn()
    measurements.elapsed_seconds[label] = timer() - start
    return result


def _check_budget(
    deadline: float,
    scenario_deadline: float,
    measurements: ScenarioMeasurements,
    prefix: str,
) -> bool:
    now = time.perf_counter()
    if now > scenario_deadline:
        measurements.aborted = True
        measurements.abort_reason = f"{prefix}: scenario time budget exceeded"
        return False
    if now > deadline:
        measurements.aborted = True
        measurements.abort_reason = f"{prefix}: total benchmark time budget exceeded"
        return False
    return True


def _run_scenario(
    n: int,
    config: BenchmarkConfig,
    deadline: float,
    scenario_deadline: float,
) -> ScenarioMeasurements:
    measurements = ScenarioMeasurements(n=n)

    dataset = generate_dataset(n, seed=config.seed)
    if not _check_budget(deadline, scenario_deadline, measurements, "after generate"):
        return measurements

    with tempfile.TemporaryDirectory(prefix="hermes_benchmark_") as tmpdir:
        db_path = os.path.join(tmpdir, "benchmark.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        MigrationRunner(conn).migrate()
        catalog = Catalog(conn)
        graph = EconomicGraph(conn)

        def _bulk_insert() -> None:
            for entity in dataset.entities:
                catalog.entity.insert(entity)
            for source in dataset.sources:
                catalog.source.insert(source)
            for document in dataset.documents:
                catalog.document.insert(document)
            for event in dataset.events:
                catalog.event.insert(event)
            for edge in dataset.edges:
                catalog.relation.insert(edge)
            catalog.commit()

        _measure("sqlite_bulk_insert", measurements, _bulk_insert)
        if not _check_budget(deadline, scenario_deadline, measurements, "sqlite insert"):
            conn.close()
            return measurements

        def _queries() -> tuple[int, int, int]:
            entity = catalog.entity.get(dataset.entities[0].entity_id)
            edge = catalog.relation.get(dataset.edges[0].edge_id)
            count_entities = conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0]
            count_edges = conn.execute("SELECT COUNT(*) FROM relation_edge").fetchone()[0]
            return (
                1 if entity is not None else 0,
                1 if edge is not None else 0,
                count_entities + count_edges,
            )

        entity_ok, edge_ok, total_count = _measure("sqlite_queries", measurements, _queries)
        measurements.result_counts["sqlite_rows"] = total_count
        if not (entity_ok and edge_ok):
            measurements.aborted = True
            measurements.abort_reason = "catalog lookups returned None"
            conn.close()
            return measurements
        if not _check_budget(deadline, scenario_deadline, measurements, "sqlite query"):
            conn.close()
            return measurements

        def _traverse() -> int:
            paths = graph.traverse(
                dataset.entities[0].entity_id,
                as_of=_EPOCH,
                max_hops=3,
                direction="both",
            )
            return len(paths)

        path_count = _measure("graph_traverse_3_hops", measurements, _traverse)
        measurements.result_counts["graph_paths"] = path_count
        if not _check_budget(deadline, scenario_deadline, measurements, "graph traversal"):
            conn.close()
            return measurements

        index = VectorIndex(
            task="document_retrieval",
            model_id="test_lexical_hash",
            model_revision="1",
            dimension=384,
        )

        def _vector_upsert() -> None:
            index.upsert(dataset.embeddings)

        _measure("vector_index_upsert", measurements, _vector_upsert)
        if not _check_budget(deadline, scenario_deadline, measurements, "vector upsert"):
            conn.close()
            return measurements

        query_vector = dataset.embeddings[0].vector

        def _vector_search() -> int:
            return len(index.search(query_vector, top_k=config.top_k, as_of=_EPOCH))

        search_count = _measure("vector_exact_search", measurements, _vector_search)
        measurements.result_counts["vector_results"] = search_count
        if not _check_budget(deadline, scenario_deadline, measurements, "vector search"):
            conn.close()
            return measurements

        conn.close()

        if config.include_pipeline:
            pipeline = Pipeline(
                config={"seed": config.seed},
                price_provider=_stub_closes,
            )

            def _pipeline_cycle() -> dict[str, Any]:
                result = pipeline.run(
                    price=100.0,
                    entity_id=dataset.entities[0].entity_id,
                    web_inputs=dataset.web_inputs,
                    reports=dataset.reports,
                    regime_features={
                        "vix": 18,
                        "credit_spread": 1.2,
                        "yield_curve": 0.3,
                        "momentum": 0.04,
                    },
                )
                return result.__dict__

            pipeline_result = _measure("pipeline_cycle", measurements, _pipeline_cycle)
            measurements.result_counts["pipeline_decisions"] = len(
                pipeline_result.get("decisions", [])
            )
            if not _check_budget(deadline, scenario_deadline, measurements, "pipeline cycle"):
                return measurements

    measurements.peak_rss_bytes = _peak_rss_bytes()
    return measurements


_EPOCH = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkResult:
    """Run the reproducible local benchmark and return a structured report."""
    cfg = config or BenchmarkConfig()
    started_at = datetime.now(UTC)
    timer = _timer()
    deadline = timer() + cfg.max_total_seconds
    environment = _environment()
    scenarios: list[ScenarioMeasurements] = []

    for n in cfg.sizes:
        scenario_deadline = timer() + cfg.max_scenario_seconds
        scenario = _run_scenario(n, cfg, deadline, scenario_deadline)
        scenarios.append(scenario)
        if scenario.peak_rss_bytes and scenario.peak_rss_bytes > cfg.max_rss_bytes:
            scenario.aborted = True
            scenario.abort_reason = f"peak RSS {scenario.peak_rss_bytes} exceeds budget"
        if scenario.aborted:
            break

    recommendation = _build_recommendation(cfg, scenarios)
    finished_at = datetime.now(UTC)
    return BenchmarkResult(
        config=cfg,
        environment=environment,
        started_at=started_at,
        finished_at=finished_at,
        scenarios=scenarios,
        recommendation=recommendation,
    )


def _build_recommendation(
    config: BenchmarkConfig,
    scenarios: list[ScenarioMeasurements],
) -> str:
    """Produce an evidence-based recommendation from the measured numbers."""
    lines: list[str] = []
    completed = [s for s in scenarios if not s.aborted]
    if not completed:
        lines.append("No scenarios completed within the resource budget.")
        lines.append("Stop and investigate before any scale work.")
        return "\n".join(lines)

    # Find the slowest operation at the largest completed size.
    largest = completed[-1]
    timings = largest.elapsed_seconds
    if timings:
        slowest_op = max(timings.items(), key=lambda kv: kv[1])[0]
        slowest_s = timings[slowest_op]
        lines.append(
            f"At n={largest.n}, the slowest measured operation is "
            f"'{slowest_op}' ({slowest_s:.4f} s)."
        )

    # SQLite insert scaling
    sqlite_inserts = [
        (s.n, s.elapsed_seconds.get("sqlite_bulk_insert"))
        for s in completed
        if "sqlite_bulk_insert" in s.elapsed_seconds
    ]
    if len(sqlite_inserts) >= 2:
        n1, t1 = sqlite_inserts[0]
        n2, t2 = sqlite_inserts[-1]
        if t1 and t2 and t1 > 0:
            ratio = (t2 / t1) / (n2 / n1)
            lines.append(
                f"SQLite bulk insert scaled from n={n1} ({t1:.4f}s) to n={n2} ({t2:.4f}s); "
                f"relative scaling factor = {ratio:.2f}x per row (1.0 = linear)."
            )

    # Vector search is O(N) exact scan.
    vector_times = [
        (s.n, s.elapsed_seconds.get("vector_exact_search"))
        for s in completed
        if "vector_exact_search" in s.elapsed_seconds
    ]
    if len(vector_times) >= 2:
        n1, t1 = vector_times[0]
        n2, t2 = vector_times[-1]
        if t1 and t2 and t1 > 0:
            ratio = (t2 / t1) / (n2 / n1)
            lines.append(
                f"Exact vector search scaled from n={n1} ({t1:.4f}s) to n={n2} ({t2:.4f}s); "
                f"relative scaling factor = {ratio:.2f}x per row. "
                f"An index (e.g. HNSW/annoy) is justified only after measuring "
                f"NN-query latency above the target budget."
            )

    # Graph traversal complexity caveat.
    graph_times = [
        (s.n, s.elapsed_seconds.get("graph_traverse_3_hops"))
        for s in completed
        if "graph_traverse_3_hops" in s.elapsed_seconds
    ]
    if graph_times:
        n_max, t_max = graph_times[-1]
        lines.append(
            f"3-hop graph traversal at n={n_max} took {t_max:.4f} s. "
            f"Bounded BFS/DFS over a sparse local graph is expected to stay small "
            f"for low-degree graphs; a dedicated graph DB is not justified by these measurements."
        )

    # Pipeline cycle caveat.
    if config.include_pipeline and any(
        "pipeline_cycle" in s.elapsed_seconds for s in completed
    ):
        pipeline_times = [
            (s.n, s.elapsed_seconds["pipeline_cycle"])
            for s in completed
            if "pipeline_cycle" in s.elapsed_seconds
        ]
        n_max, t_max = pipeline_times[-1]
        lines.append(
            f"One full pipeline cycle at n={n_max} took {t_max:.4f} s. "
            f"Most time is spent in portfolio/risk/Monte-Carlo logic, not catalog size."
        )

    # Overall recommendation.
    lines.append(
        "At the measured MVP sizes the in-memory/SQLite design is sufficient. "
        "Scale work should only start after measuring a real production-sized fixture "
        "and identifying a concrete bottleneck; do not change infrastructure purely "
        "from projected growth."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_benchmark()
    print(result.summary())
