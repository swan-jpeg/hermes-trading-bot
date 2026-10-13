"""Tests for versioned document chunking and mock vector retrieval (T06)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from hermes_bot.retrieval import (
    TEST_MODEL_ID,
    Chunker,
    DimensionMismatch,
    IncompatibleIndex,
    TestEmbeddingProvider,
    VectorIndex,
    ZeroVectorRejected,
    available_providers,
    chunk_id_from,
)
from hermes_bot.schemas import Chunk, EmbeddingRecord

# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def test_chunker_empty_document() -> None:
    chunker = Chunker()
    assert chunker.chunk("doc_" + "a" * 64, "") == []


def test_chunker_preserves_exact_offsets() -> None:
    doc_id = "doc_" + "b" * 64
    text = "First sentence here. Second sentence follows. Third one ends."
    chunker = Chunker(max_tokens=100, overlap_tokens=5)
    chunks = chunker.chunk(doc_id, text)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_offset == 0
    assert chunk.end_offset == len(text)
    assert text[chunk.start_offset : chunk.end_offset] == text


def test_chunker_unicode_not_split() -> None:
    doc_id = "doc_" + "c" * 64
    text = "\U0001f600 emoji. \u00e9\u00e8 accented."
    chunker = Chunker(max_tokens=100)
    chunks = chunker.chunk(doc_id, text)
    for chunk in chunks:
        assert chunk.text == text[chunk.start_offset : chunk.end_offset]
        # Round-trip through slicing proves no code-point split.
        sliced = text[chunk.start_offset : chunk.end_offset]
        assert chunk.text.encode("utf-8") == sliced.encode("utf-8")


def test_chunker_overlap_and_version() -> None:
    doc_id = "doc_" + "d" * 64
    sentences = [f"Sentence number {i} has a few words in it." for i in range(20)]
    text = " ".join(sentences)
    chunker = Chunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk(doc_id, text)
    assert len(chunks) >= 2
    assert all(c.chunker_version == "paragraph_sentence_v1" for c in chunks)

    # Adjacent chunks must overlap in text (not just whitespace).
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.end_offset > nxt.start_offset
        overlap = text[nxt.start_offset : prev.end_offset].strip()
        overlap_tokens = len(overlap.split())
        assert overlap_tokens >= 5


def test_chunker_does_not_cross_paragraphs() -> None:
    doc_id = "doc_" + "e" * 64
    text = "Para one sentence.\n\nPara two sentence."
    chunker = Chunker(max_tokens=100)
    chunks = chunker.chunk(doc_id, text)
    assert len(chunks) == 2
    assert chunks[0].end_offset <= text.index("Para two")
    assert chunks[1].start_offset >= text.index("Para two")


def test_chunk_id_is_deterministic_and_sensitive() -> None:
    doc_id = "doc_" + "f" * 64
    id1 = chunk_id_from(doc_id, "paragraph_sentence_v1", 0, 10, "hello")
    id2 = chunk_id_from(doc_id, "paragraph_sentence_v1", 0, 10, "hello")
    id3 = chunk_id_from(doc_id, "paragraph_sentence_v1", 0, 10, "world")
    assert id1 == id2
    assert id1 != id3
    assert id1.startswith("chk_")
    assert len(id1) == 4 + 64


def test_chunk_model_rejects_invalid_chunk_id() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="bad_id",
            document_id="doc_" + "g" * 64,
            chunker_version="paragraph_sentence_v1",
            start_offset=0,
            end_offset=5,
            text="hello",
            created_at=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_provider_is_test_only_and_not_selectable() -> None:
    provider = TestEmbeddingProvider()
    assert provider.test_only is True
    assert TEST_MODEL_ID not in available_providers()


def test_provider_dimension_and_count() -> None:
    provider = TestEmbeddingProvider()
    records = provider.embed(["one", "two"], {"object_id": "obj_1"})
    assert len(records) == 2
    for record in records:
        assert record.dimension == 384
        assert len(record.vector) == 384
        assert record.normalized is True
        assert all(isinstance(v, float) and not (v != v) for v in record.vector)


def test_provider_is_deterministic() -> None:
    provider = TestEmbeddingProvider()
    r1 = provider.embed(["hello world"], {"object_id": "obj_1"})[0]
    r2 = provider.embed(["hello world"], {"object_id": "obj_2"})[0]
    assert r1.vector == r2.vector
    assert r1.model_id == "test_lexical_hash"


def test_provider_per_text_metadata() -> None:
    provider = TestEmbeddingProvider()
    records = provider.embed(
        ["a", "b"],
        [{"object_id": "obj_a", "section": "intro"}, {"object_id": "obj_b", "section": "outro"}],
    )
    assert records[0].object_id == "obj_a"
    assert records[0].metadata["section"] == "intro"
    assert records[1].object_id == "obj_b"
    assert records[1].metadata["section"] == "outro"


def test_provider_vector_passes_schema_validation() -> None:
    provider = TestEmbeddingProvider()
    record = provider.embed(["x"], {"object_id": "obj_x"})[0]
    assert isinstance(record, EmbeddingRecord)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def _make_index() -> VectorIndex:
    return VectorIndex(
        task="document_retrieval",
        model_id="test_lexical_hash",
        model_revision="1",
        dimension=384,
    )


def _record_for(text: str, object_id: str, available_at: datetime | None = None) -> EmbeddingRecord:
    provider = TestEmbeddingProvider()
    return provider.embed(
        [text],
        {"object_id": object_id, "start_offset": 0, "end_offset": len(text)},
        available_at=available_at,
    )[0]


def test_index_rejects_zero_vector() -> None:
    index = _make_index()
    record = _record_for("hello", "obj_1")
    # Mutate vector to zero after creation.
    bad = record.model_copy(update={"vector": [0.0] * 384})
    with pytest.raises(ZeroVectorRejected):
        index.upsert([bad])


def test_index_rejects_non_finite_vector() -> None:
    index = _make_index()
    record = _record_for("hello", "obj_1")
    bad = record.model_copy(update={"vector": [float("inf")] + ([0.0] * 383)})
    with pytest.raises(ZeroVectorRejected):
        index.upsert([bad])


def test_index_rejects_dimension_mismatch() -> None:
    index = _make_index()
    record = _record_for("hello", "obj_1")
    bad = record.model_copy(update={"dimension": 128, "vector": [0.1] * 128})
    with pytest.raises(DimensionMismatch):
        index.upsert([bad])


def test_index_rejects_model_version_mismatch() -> None:
    index = _make_index()
    record = _record_for("hello", "obj_1")
    bad = record.model_copy(update={"model_revision": "2"})
    with pytest.raises(IncompatibleIndex):
        index.upsert([bad])


def test_index_duplicate_upsert_is_deterministic() -> None:
    index = _make_index()
    record = _record_for("hello", "obj_1")
    index.upsert([record])
    assert len(index) == 1
    # Upserting the same embedding_id again keeps one record.
    index.upsert([record])
    assert len(index) == 1
    # A different vector for the same id overwrites the previous vector.
    other = record.model_copy(update={"vector": _record_for("world", "obj_1").vector})
    index.upsert([other])
    assert len(index) == 1
    # Score should now be 1.0 for the new vector.
    results = index.search(other.vector)
    assert results[0].score == pytest.approx(1.0)


def test_index_ranking_exact_cosine() -> None:
    index = _make_index()
    r1 = _record_for("alpha beta gamma", "obj_1")
    r2 = _record_for("delta epsilon zeta", "obj_2")
    r3 = _record_for("alpha beta gamma", "obj_3")
    index.upsert([r1, r2, r3])

    query = _record_for("alpha beta gamma", "query").vector
    results = index.search(query, top_k=3)
    assert len(results) == 3
    assert results[0].object_id in {"obj_1", "obj_3"}
    assert results[0].score == pytest.approx(1.0)
    # Identical texts should tie exactly.
    assert results[1].score == pytest.approx(1.0)
    assert results[2].object_id == "obj_2"
    # All scores are cosine similarities in [-1, 1].
    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_index_returns_source_offsets() -> None:
    index = _make_index()
    record = _record_for("hello world", "obj_1")
    assert record.metadata["start_offset"] == 0
    assert record.metadata["end_offset"] == len("hello world")
    index.upsert([record])
    results = index.search(record.vector)
    assert results[0].source_offsets == (0, len("hello world"))


def test_index_availability_filter() -> None:
    index = _make_index()
    t0 = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(hours=1)
    r1 = _record_for("early", "obj_1", available_at=t0)
    r2 = _record_for("late", "obj_2", available_at=t1)
    index.upsert([r1, r2])

    query = _record_for("early", "query").vector
    results = index.search(query, as_of=t0)
    assert len(results) == 1
    assert results[0].object_id == "obj_1"

    results = index.search(query, as_of=t1)
    assert len(results) == 2


def test_index_metadata_filter() -> None:
    index = _make_index()
    r1 = _record_for("foo bar", "obj_1")
    r1 = r1.model_copy(update={"metadata": {**r1.metadata, "section": "intro"}})
    r2 = _record_for("foo bar", "obj_2")
    r2 = r2.model_copy(update={"metadata": {**r2.metadata, "section": "outro"}})
    index.upsert([r1, r2])

    query = _record_for("foo bar", "query").vector
    results = index.search(query, metadata_filter={"section": "intro"})
    assert len(results) == 1
    assert results[0].object_id == "obj_1"


def test_index_query_dimension_mismatch() -> None:
    index = _make_index()
    with pytest.raises(DimensionMismatch):
        index.search([0.1] * 128)


def test_index_empty_search_returns_empty() -> None:
    index = _make_index()
    query = _record_for("anything", "query").vector
    assert index.search(query) == []
