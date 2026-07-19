"""
===============================================================================
ingestion_chunker.py
===============================================================================
Build deterministic, structure-aware chunks from preprocessed PDF content.

Responsibilities:
  - Traverse preprocessed document sources in stable order.
  - Split text with explicit character overlap and emit canonical metadata.
  - Validate the chunk contract shared with embeddings and FAISS.

Design principles:
  - Make every identity and character-based length unit traceable.
  - Derive stable chunk identifiers from document and source positions.

Boundaries:
  - Does not parse PDFs, create embeddings, or persist vectors.
  - Does not claim retrieval-quality improvements from its structure rules.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping

__all__ = [
    "CHUNK_LENGTH_UNIT",
    "CHUNK_SCHEMA_VERSION",
    "SUPPORTED_SOURCE_TYPES",
    "ChunkingError",
    "InvalidChunkError",
    "InvalidDocumentError",
    "PDFChunker",
]


# Public schema values shared by ingestion, embeddings, and persisted records.
CHUNK_SCHEMA_VERSION = 1
CHUNK_LENGTH_UNIT = "characters"
SUPPORTED_SOURCE_TYPES = {
    "paragraph",
    "heading",
    "caption",
    "pseudo_table",
    "table",
}


class ChunkingError(ValueError):
    """Represent a UI-safe document or chunk-schema validation failure."""


class InvalidDocumentError(ChunkingError):
    """Indicate that loader or preprocessor output cannot be chunked safely."""


class InvalidChunkError(ChunkingError):
    """Indicate that a chunk violates the shared ingestion and storage schema."""


class PDFChunker:
    """Create traceable PDF chunks with character-based length and overlap.

    Parameters
    ----------
    max_chunk_length
        Positive maximum number of Unicode code points per emitted chunk.
    overlap_length
        Non-negative code-point overlap, strictly smaller than the maximum.

    Notes
    -----
    Sources are traversed in deterministic structural order. Every chunk records
    the shared schema version, ``characters`` length unit, content-derived
    document identity, source position, part position, and source metadata.
    """

    def __init__(
        self,
        max_chunk_length: int = 1000,
        overlap_length: int = 200,
    ) -> None:
        """Configure deterministic character length and overlap bounds."""

        if not isinstance(max_chunk_length, int) or max_chunk_length <= 0:
            raise ValueError("max_chunk_length must be a positive integer")
        if not isinstance(overlap_length, int):
            raise ValueError("overlap_length must be an integer")
        if not 0 <= overlap_length < max_chunk_length:
            raise ValueError(
                "overlap_length must satisfy 0 <= overlap_length < " "max_chunk_length"
            )
        self.max_chunk_length = max_chunk_length
        self.overlap_length = overlap_length

    def split_text(self, text: str) -> list[str]:
        """Split text deterministically within the configured character bound.

        Parameters
        ----------
        text
            Source text to normalize and split.

        Returns
        -------
        list of str
            Non-empty parts whose lengths do not exceed ``max_chunk_length``.

        Raises
        ------
        InvalidDocumentError
            If ``text`` is not a string.
        """

        if not isinstance(text, str):
            raise InvalidDocumentError("Source text must be a string.")
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.max_chunk_length:
            return [text]

        step = self.max_chunk_length - self.overlap_length
        parts: list[str] = []
        start = 0
        while start < len(text):
            part = text[start : start + self.max_chunk_length]
            if part:
                parts.append(part)
            if start + self.max_chunk_length >= len(text):
                break
            start += step
        return parts

    def chunk_document(self, document: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Chunk a preprocessed document in stable structural order.

        Parameters
        ----------
        document
            Loader/preprocessor mapping with a SHA-256 document identity and
            supported structural source collections.

        Returns
        -------
        list of dict
            Canonical chunks ordered by source and part sequence.

        Raises
        ------
        InvalidDocumentError
            If required document metadata or structural fields are invalid.
        InvalidChunkError
            If an emitted record violates the shared chunk schema.
        """

        if not isinstance(document, Mapping):
            raise InvalidDocumentError("Document must be a mapping.")
        metadata = document.get("metadata")
        pages = document.get("pages")
        if not isinstance(metadata, dict):
            raise InvalidDocumentError("Document metadata must be a dictionary.")
        if not isinstance(pages, list):
            raise InvalidDocumentError("Document pages must be a list.")

        document_id = self._document_id(document)
        document_metadata = self._document_metadata(metadata, document_id)
        tables_by_page, tables_without_page = self._tables_by_page(metadata)

        chunks: list[dict[str, Any]] = []
        chunk_sequence = 0
        source_sequence = 0

        for page_position, page in enumerate(pages):
            if not isinstance(page, Mapping):
                raise InvalidDocumentError(
                    f"Page at position {page_position} must be a mapping."
                )
            page_number = page.get("page", page_position + 1)
            if not isinstance(page_number, int) or page_number <= 0:
                raise InvalidDocumentError(
                    f"Page at position {page_position} has an invalid page number."
                )
            paragraphs = page.get("paragraphs", [])
            if not isinstance(paragraphs, list):
                raise InvalidDocumentError(
                    f"Paragraphs on page {page_number} must be a list."
                )

            for paragraph_index, paragraph in enumerate(paragraphs):
                if not isinstance(paragraph, Mapping):
                    raise InvalidDocumentError(
                        f"Paragraph {paragraph_index} on page {page_number} must be "
                        "a mapping."
                    )
                source_type = self._paragraph_source_type(paragraph)
                source_metadata = {
                    "page_number": page_number,
                    "paragraph_index": paragraph_index,
                    "heading_level": paragraph.get("heading_level", 0),
                    "text_hash": paragraph.get("text_hash"),
                }
                emitted = self._emit_source_chunks(
                    text=paragraph.get("text"),
                    source_type=source_type,
                    source_sequence=source_sequence,
                    chunk_sequence_start=chunk_sequence,
                    document_metadata=document_metadata,
                    source_metadata=source_metadata,
                )
                chunks.extend(emitted)
                chunk_sequence += len(emitted)
                source_sequence += 1

            for table_index, table in tables_by_page.get(page_number, []):
                emitted = self._emit_table_chunks(
                    table=table,
                    table_index=table_index,
                    page_number=page_number,
                    source_sequence=source_sequence,
                    chunk_sequence_start=chunk_sequence,
                    document_metadata=document_metadata,
                )
                chunks.extend(emitted)
                chunk_sequence += len(emitted)
                source_sequence += 1

        for table_index, table in tables_without_page:
            emitted = self._emit_table_chunks(
                table=table,
                table_index=table_index,
                page_number=None,
                source_sequence=source_sequence,
                chunk_sequence_start=chunk_sequence,
                document_metadata=document_metadata,
            )
            chunks.extend(emitted)
            chunk_sequence += len(emitted)
            source_sequence += 1

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise InvalidChunkError("Chunk generation produced duplicate IDs.")
        for chunk in chunks:
            self.validate_chunk(chunk)
        return chunks

    @staticmethod
    def validate_chunk(chunk: Mapping[str, Any]) -> None:
        """Validate one chunk against the shared embedding and storage schema.

        Parameters
        ----------
        chunk
            Candidate chunk mapping.

        Raises
        ------
        InvalidChunkError
            If identity, text, metadata, length unit, or source fields are invalid.
        """

        if not isinstance(chunk, Mapping):
            raise InvalidChunkError("Chunk must be a mapping.")
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        metadata = chunk.get("metadata")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise InvalidChunkError("chunk_id must be a non-empty string.")
        if not isinstance(text, str) or not text:
            raise InvalidChunkError(f"Chunk {chunk_id!r} text must be non-empty.")
        if not isinstance(metadata, dict):
            raise InvalidChunkError(
                f"Chunk {chunk_id!r} metadata must be a dictionary."
            )

        required_metadata = {
            "schema_version": int,
            "length_unit": str,
            "document_id": str,
            "document_title": str,
            "source_type": str,
            "source_sequence": int,
            "chunk_sequence": int,
            "part_index": int,
            "part_count": int,
        }
        for name, expected_type in required_metadata.items():
            value = metadata.get(name)
            valid_type = (
                type(value) is int
                if expected_type is int
                else isinstance(value, expected_type)
            )
            if not valid_type:
                raise InvalidChunkError(
                    f"Chunk {chunk_id!r} metadata field {name!r} must be "
                    f"{expected_type.__name__}."
                )
            if expected_type is str and (
                not isinstance(value, str) or not value.strip()
            ):
                raise InvalidChunkError(
                    f"Chunk {chunk_id!r} metadata field {name!r} must be non-empty."
                )
        if metadata["schema_version"] != CHUNK_SCHEMA_VERSION:
            raise InvalidChunkError(
                f"Chunk {chunk_id!r} has an unsupported schema version."
            )
        if metadata["length_unit"] != CHUNK_LENGTH_UNIT:
            raise InvalidChunkError(f"Chunk {chunk_id!r} has an invalid length unit.")
        if metadata["source_type"] not in SUPPORTED_SOURCE_TYPES:
            raise InvalidChunkError(f"Chunk {chunk_id!r} has an invalid source type.")
        for name in ("source_sequence", "chunk_sequence", "part_index"):
            if metadata[name] < 0:
                raise InvalidChunkError(
                    f"Chunk {chunk_id!r} metadata field {name!r} must be non-negative."
                )
        if not 0 <= metadata["part_index"] < metadata["part_count"]:
            raise InvalidChunkError(f"Chunk {chunk_id!r} has invalid part indices.")

    def _emit_source_chunks(
        self,
        *,
        text: Any,
        source_type: str,
        source_sequence: int,
        chunk_sequence_start: int,
        document_metadata: Mapping[str, Any],
        source_metadata: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise InvalidDocumentError(
                f"{source_type} source {source_sequence} text must be a string."
            )
        parts = self.split_text(text)
        emitted: list[dict[str, Any]] = []
        for part_index, part in enumerate(parts):
            chunk_sequence = chunk_sequence_start + part_index
            chunk_id = self._chunk_id(
                document_metadata["document_id"],
                chunk_sequence,
                source_type,
                part_index,
            )
            metadata = {
                **document_metadata,
                **source_metadata,
                "schema_version": CHUNK_SCHEMA_VERSION,
                "length_unit": CHUNK_LENGTH_UNIT,
                "source_type": source_type,
                "source_sequence": source_sequence,
                "chunk_sequence": chunk_sequence,
                "part_index": part_index,
                "part_count": len(parts),
            }
            emitted.append({"chunk_id": chunk_id, "text": part, "metadata": metadata})
        return emitted

    def _emit_table_chunks(
        self,
        *,
        table: Mapping[str, Any],
        table_index: int,
        page_number: int | None,
        source_sequence: int,
        chunk_sequence_start: int,
        document_metadata: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        rows = table.get("table")
        if not isinstance(rows, list):
            raise InvalidDocumentError(f"Table {table_index} rows must be a list.")
        serialised_rows: list[str] = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                raise InvalidDocumentError(
                    f"Row {row_index} in table {table_index} must be a list."
                )
            serialised_rows.append(
                " | ".join("" if cell is None else str(cell) for cell in row)
            )
        return self._emit_source_chunks(
            text="\n".join(serialised_rows),
            source_type="table",
            source_sequence=source_sequence,
            chunk_sequence_start=chunk_sequence_start,
            document_metadata=document_metadata,
            source_metadata={
                "page_number": page_number,
                "table_index": table_index,
            },
        )

    @staticmethod
    def _paragraph_source_type(paragraph: Mapping[str, Any]) -> str:
        paragraph_type = paragraph.get("is_type", "normal")
        heading_level = paragraph.get("heading_level", 0)
        if isinstance(heading_level, int) and heading_level > 0:
            return "heading"
        if paragraph_type in {"caption", "pseudo_table"}:
            return paragraph_type
        return "paragraph"

    @staticmethod
    def _tables_by_page(
        metadata: Mapping[str, Any],
    ) -> tuple[
        dict[int, list[tuple[int, Mapping[str, Any]]]],
        list[tuple[int, Mapping[str, Any]]],
    ]:
        raw_tables = metadata.get("tables", [])
        if not isinstance(raw_tables, list):
            raise InvalidDocumentError("metadata.tables must be a list.")
        by_page: dict[int, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
        without_page: list[tuple[int, Mapping[str, Any]]] = []
        for table_index, table in enumerate(raw_tables):
            if not isinstance(table, Mapping):
                raise InvalidDocumentError(
                    f"Table at position {table_index} must be a mapping."
                )
            page_number = table.get("page")
            if isinstance(page_number, int) and page_number > 0:
                by_page[page_number].append((table_index, table))
            else:
                without_page.append((table_index, table))
        return dict(by_page), without_page

    @staticmethod
    def _document_id(document: Mapping[str, Any]) -> str:
        metadata = document["metadata"]
        file_hash = metadata.get("file_hash")
        if isinstance(file_hash, str) and file_hash.strip():
            return file_hash.strip().lower()
        canonical_document = json.dumps(
            document, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(canonical_document).hexdigest()

    @staticmethod
    def _document_metadata(
        metadata: Mapping[str, Any], document_id: str
    ) -> dict[str, Any]:
        title = metadata.get("document_title") or metadata.get("file_name")
        if not isinstance(title, str) or not title.strip():
            title = "Untitled document"
        result: dict[str, Any] = {
            "document_id": document_id,
            "document_title": title.strip(),
        }
        for field in ("file_name", "document_language", "author", "subject"):
            value = metadata.get(field)
            if value is None or isinstance(value, (str, int, float, bool)):
                result[field] = value
        return result

    @staticmethod
    def _chunk_id(
        document_id: str,
        chunk_sequence: int,
        source_type: str,
        part_index: int,
    ) -> str:
        return (
            f"{document_id}:{chunk_sequence:06d}:{source_type}:"
            f"part-{part_index:04d}"
        )
