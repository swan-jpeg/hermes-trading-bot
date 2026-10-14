"""CI-safe smoke test for the benchmark harness (T17).

Keeps the benchmark itself out of the normal unit-test suite while verifying
that the generator, runner, and all measured code paths run deterministically
on a tiny in-memory dataset.
"""
from __future__ import annotations

import pytest

from hermes_bot.benchmark.generator import generate_dataset
from hermes_bot.benchmark.runner import BenchmarkConfig, run_benchmark


def _dump_list(records: list) -> list[dict]:
    """Canonical dict dump for pydantic records."""
    return [r.model_dump() for r in records]


def test_generator_is_deterministic() -> None:
    """Same seed produces a fully identical dataset, including embeddings."""
    a = generate_dataset(10, seed=7)
    b = generate_dataset(10, seed=7)
    assert a.n == b.n == 10
    assert _dump_list(a.entities) == _dump_list(b.entities)
    assert _dump_list(a.sources) == _dump_list(b.sources)
    assert _dump_list(a.documents) == _dump_list(b.documents)
    assert _dump_list(a.events) == _dump_list(b.events)
    assert _dump_list(a.edges) == _dump_list(b.edges)
    assert _dump_list(a.embeddings) == _dump_list(b.embeddings)
    assert a.web_inputs == b.web_inputs
    assert a.reports == b.reports
    assert len(a.embeddings) == len(b.embeddings) == 10
    assert (
        [e.embedding_id for e in a.embeddings]
        == [e.embedding_id for e in b.embeddings]
    )


def test_generator_differs_across_seeds() -> None:
    """A different seed produces a different dataset (no constant output)."""
    a = generate_dataset(10, seed=7)
    c = generate_dataset(10, seed=8)
    assert [e.entity_id for e in a.entities] != [e.entity_id for e in c.entities]
    assert _dump_list(a.embeddings) != _dump_list(c.embeddings)
    assert (
        [e.embedding_id for e in a.embeddings]
        != [e.embedding_id for e in c.embeddings]
    )


def test_generator_bounds_size() -> None:
    """Generator rejects non-positive sizes."""
    with pytest.raises(ValueError):
        generate_dataset(0)
    with pytest.raises(ValueError):
        generate_dataset(-5)


def test_benchmark_smoke_runs_in_memory() -> None:
    """The full benchmark runs for 10 records without touching persistent storage."""
    config = BenchmarkConfig(
        sizes=(10,),
        seed=123,
        max_total_seconds=30.0,
        max_scenario_seconds=10.0,
        include_pipeline=True,
    )
    result = run_benchmark(config)
    assert len(result.scenarios) == 1
    scenario = result.scenarios[0]
    assert not scenario.aborted, scenario.abort_reason
    assert scenario.n == 10

    # Every measured component produced a result.
    assert "sqlite_bulk_insert" in scenario.elapsed_seconds
    assert "sqlite_queries" in scenario.elapsed_seconds
    assert "graph_traverse_3_hops" in scenario.elapsed_seconds
    assert "vector_index_upsert" in scenario.elapsed_seconds
    assert "vector_exact_search" in scenario.elapsed_seconds
    assert "pipeline_cycle" in scenario.elapsed_seconds

    # Sanity counts.
    assert scenario.result_counts["sqlite_rows"] >= 20  # 10 entities + >=10 edges
    assert scenario.result_counts["graph_paths"] >= 1
    assert scenario.result_counts["vector_results"] >= 1
    assert scenario.result_counts["pipeline_decisions"] >= 0

    # Recommendation is evidence-based and does not invent scale claims.
    assert "in-memory/SQLite" in result.recommendation or "measured" in result.recommendation
    assert "million" not in result.recommendation.lower()


def test_benchmark_smoke_without_pipeline() -> None:
    """The benchmark can skip the pipeline cycle."""
    config = BenchmarkConfig(
        sizes=(10,),
        seed=456,
        include_pipeline=False,
    )
    result = run_benchmark(config)
    assert not result.scenarios[0].aborted
    assert "pipeline_cycle" not in result.scenarios[0].elapsed_seconds
