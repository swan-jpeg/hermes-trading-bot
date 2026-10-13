"""Versioned document chunker with paragraph-then-sentence splitting (T06).

The chunker keeps exact character offsets into the original document, never
splits a Unicode code point, and produces chunks of at most 400 whitespace
tokens with a 40-token overlap between adjacent chunks.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from hermes_bot.schemas import Chunk

CHUNKER_VERSION = "paragraph_sentence_v1"
MAX_TOKENS = 400
OVERLAP_TOKENS = 40

# Sentence boundary: . ! ? followed by whitespace or end of string.
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+(?=\s*$)")


@dataclass(frozen=True)
class _Sentence:
    text: str
    start: int
    end: int

    @property
    def token_count(self) -> int:
        return len(self.text.split())


def _split_sentences(text: str) -> list[_Sentence]:
    """Split text into sentences while preserving exact character offsets."""
    sentences: list[_Sentence] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        start = match.start()
        end = match.end()
        # Strip trailing whitespace from the sentence text but keep offsets
        # anchored to the non-whitespace span so adjacent chunks overlap in
        # actual content, not in trailing spaces.
        stripped = raw.rstrip()
        trailing = len(raw) - len(stripped)
        sentences.append(_Sentence(stripped, start, end - trailing))
    return sentences


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    """Return non-empty paragraph (start, end) offsets."""
    paragraphs: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.start()
        if end > start:
            paragraphs.append((start, end))
        start = match.end()
    if start < len(text):
        paragraphs.append((start, len(text)))
    return paragraphs


def chunk_id_from(document_id: str, chunker_version: str, start: int, end: int, text: str) -> str:
    """Deterministic chunk ID from document identity, chunker version and offsets."""
    payload = {
        "document_id": document_id,
        "chunker_version": chunker_version,
        "start_offset": start,
        "end_offset": end,
        "text": text,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return "chk_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Chunker:
    """Split documents into overlapping paragraph/sentence chunks.

    Parameters
    ----------
    max_tokens:
        Maximum number of whitespace-split tokens per chunk.
    overlap_tokens:
        Minimum overlap in tokens between consecutive chunks.
    chunker_version:
        Immutable version string stored in every chunk.
    """

    def __init__(
        self,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
        chunker_version: str = CHUNKER_VERSION,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chunker_version = chunker_version

    def chunk(
        self,
        document_id: str,
        text: str,
        section: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Chunk ``text`` into ``Chunk`` records with exact offsets.

        Paragraph boundaries are honoured: a chunk never crosses paragraphs.
        Within a paragraph sentences are greedily packed up to ``max_tokens``.
        The next chunk starts with enough trailing sentences to cover
        ``overlap_tokens`` tokens.
        """
        if not text:
            return []

        created_at = datetime.now(UTC)
        chunks: list[Chunk] = []

        for para_start, para_end in _split_paragraphs(text):
            para_text = text[para_start:para_end]
            sentences = _split_sentences(para_text)
            if not sentences:
                continue

            # Work with sentence offsets relative to the whole document.
            abs_sentences = [
                _Sentence(s.text, para_start + s.start, para_start + s.end)
                for s in sentences
            ]

            current_start = 0
            while current_start < len(abs_sentences):
                # Greedy pack sentences from current_start.
                token_count = 0
                idx = current_start
                while idx < len(abs_sentences):
                    next_count = token_count + abs_sentences[idx].token_count
                    if idx > current_start and next_count > self.max_tokens:
                        break
                    token_count = next_count
                    idx += 1

                first = abs_sentences[current_start]
                last = abs_sentences[idx - 1]
                chunk_text = text[first.start : last.end]

                chunk_id = chunk_id_from(
                    document_id, self.chunker_version, first.start, last.end, chunk_text
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunker_version=self.chunker_version,
                        start_offset=first.start,
                        end_offset=last.end,
                        text=chunk_text,
                        section=section,
                        metadata=metadata or {},
                        created_at=created_at,
                    )
                )

                if idx >= len(abs_sentences):
                    break

                # Determine next start: move backward from idx until we have at
                # least overlap_tokens in the overlap region, but always advance.
                overlap_tokens = 0
                next_start = idx
                while next_start > current_start and overlap_tokens < self.overlap_tokens:
                    next_start -= 1
                    overlap_tokens += abs_sentences[next_start].token_count

                if next_start <= current_start:
                    # Force progress when a single sentence exceeds the limit.
                    next_start = current_start + 1

                current_start = next_start

        return chunks
