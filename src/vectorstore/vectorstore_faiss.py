"""
===============================================================================
vectorstore_faiss.py
===============================================================================
Store validated vectors in FAISS with complete versioned snapshots.

Responsibilities:
  - Validate and index canonical embedded chunks in positional order.
  - Search defensively and persist complete snapshot generations.
  - Reject corrupt, dimensionally incompatible, or model-incompatible state.

Design principles:
  - Validate candidate state before atomic current-pointer replacement.
  - Keep record order aligned exactly with FAISS vector positions.

Boundaries:
  - Owns vector records and snapshots, not embedding or query creation.
  - Does not share stores across application sessions unless explicitly injected.
===============================================================================
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import faiss
import numpy as np

from src import ingestion

__all__ = [
    "CorruptSnapshotError",
    "DimensionMismatchError",
    "DuplicateChunkIDError",
    "FAISSStore",
    "FAISSStoreError",
    "IncompatibleSnapshotError",
    "InvalidVectorRecordError",
    "SnapshotNotFoundError",
]


SNAPSHOT_SCHEMA_VERSION = 1
INDEX_FILENAME = "index.faiss"
MANIFEST_FILENAME = "manifest.json"
CURRENT_FILENAME = "CURRENT"


class FAISSStoreError(RuntimeError):
    """Represent a project-owned FAISS validation or persistence failure."""


class SnapshotNotFoundError(FAISSStoreError):
    """Indicate that no current persisted snapshot can be selected."""


class CorruptSnapshotError(FAISSStoreError):
    """Indicate that snapshot files or index-to-record mappings are inconsistent."""


class IncompatibleSnapshotError(FAISSStoreError):
    """Indicate that persisted schema or embedding-model metadata is incompatible."""


class DuplicateChunkIDError(FAISSStoreError):
    """Indicate that chunk identities would make FAISS positions ambiguous."""


class DimensionMismatchError(FAISSStoreError):
    """Indicate that a vector or snapshot dimension differs from the store."""


class InvalidVectorRecordError(FAISSStoreError):
    """Indicate that an embedded chunk or query vector is unusable."""


class FAISSStore:
    """Store vectors and records in an unambiguous positional mapping.

    Parameters
    ----------
    snapshot_directory
        Optional directory for complete immutable snapshot generations.
    dimension
        Positive fixed vector dimension for the FAISS index.
    embedding_model
        Non-empty model identifier stored and validated with each snapshot.

    Notes
    -----
    A persistent store selects immutable generations through an atomically replaced
    ``CURRENT`` pointer. Each manifest contains every record needed to interpret
    FAISS positions. An omitted directory creates a session-local in-memory store.
    """

    def __init__(
        self,
        snapshot_directory: str | Path | None = None,
        *,
        dimension: int = 1536,
        embedding_model: str = "not-configured",
    ) -> None:
        """Create an empty store or load a validated current snapshot."""

        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(embedding_model, str) or not embedding_model.strip():
            raise ValueError("embedding_model must be a non-empty string")

        self.snapshot_directory = (
            Path(snapshot_directory) if snapshot_directory is not None else None
        )
        self.dimension = dimension
        self.embedding_model = embedding_model
        self.index = faiss.IndexFlatL2(dimension)
        self._records: list[dict[str, Any]] = []
        self._records_by_id: dict[str, dict[str, Any]] = {}

        if self.snapshot_directory is not None:
            current_path = self.snapshot_directory / CURRENT_FILENAME
            if current_path.exists():
                self.load_snapshot()
            elif self.snapshot_directory.exists() and any(
                self.snapshot_directory.iterdir()
            ):
                raise CorruptSnapshotError(
                    f"Persisted store at {self.snapshot_directory} has no CURRENT "
                    "pointer. Remove the incomplete prototype artifact or restore a "
                    "complete snapshot."
                )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return defensive copies of records in their FAISS position order."""

        return tuple(copy.deepcopy(record) for record in self._records)

    @property
    def record_count(self) -> int:
        """Return the number of indexed vector records."""

        return len(self._records)

    def add_embedded_chunks(self, embedded_chunks: Iterable[Mapping[str, Any]]) -> int:
        """Validate and atomically add embedded chunks to the active store.

        Parameters
        ----------
        embedded_chunks
            Canonical chunks containing numeric fixed-dimension embeddings.

        Returns
        -------
        int
            Number of newly indexed chunks.

        Raises
        ------
        InvalidVectorRecordError
            If a chunk or embedding violates the shared storage contract.
        DimensionMismatchError
            If an embedding does not match the configured dimension.
        DuplicateChunkIDError
            If a new identity is duplicated within the batch or active store.
        FAISSStoreError
            If configured snapshot persistence cannot complete safely.

        Notes
        -----
        Candidate index and record state replace active memory only after complete
        validation and, when configured, successful snapshot publication.
        """

        vectors, new_records = self._normalise_embedded_chunks(embedded_chunks)
        if not new_records:
            return 0

        existing_ids = set(self._records_by_id)
        new_ids = [record["chunk_id"] for record in new_records]
        new_id_counts = Counter(new_ids)
        duplicate_ids = sorted(
            chunk_id
            for chunk_id, count in new_id_counts.items()
            if count > 1 or chunk_id in existing_ids
        )
        if duplicate_ids:
            raise DuplicateChunkIDError(
                f"Chunk IDs must be unique; duplicates: {duplicate_ids}"
            )

        candidate_index = faiss.clone_index(self.index)
        candidate_index.add(vectors)
        candidate_records = [*self._records, *new_records]
        self._validate_index_and_records(candidate_index, candidate_records)

        if self.snapshot_directory is not None:
            self._write_atomic_snapshot(candidate_index, candidate_records)

        self.index = candidate_index
        self._set_records(candidate_records)
        return len(new_records)

    def search(self, query_embedding: Sequence[float], k: int = 3) -> list[dict]:
        """Return up to ``k`` nearest records with source metadata and distance.

        Parameters
        ----------
        query_embedding
            Finite numeric query vector matching the store dimension.
        k
            Non-negative maximum number of nearest records.

        Returns
        -------
        list of dict
            Defensive record copies ordered by ascending squared L2 distance.

        Raises
        ------
        ValueError
            If ``k`` is not a non-negative integer.
        DimensionMismatchError
            If the query vector has the wrong shape or dimension.
        InvalidVectorRecordError
            If the query contains non-numeric or non-finite values.
        CorruptSnapshotError
            If FAISS returns a position without a corresponding record.
        """

        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer")

        try:
            query = np.asarray(query_embedding, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise InvalidVectorRecordError(
                "Query embedding must contain numeric values."
            ) from exc
        if query.ndim != 1 or query.shape[0] != self.dimension:
            actual = query.shape[0] if query.ndim == 1 else tuple(query.shape)
            raise DimensionMismatchError(
                f"Query embedding dimension {actual!r} does not match store "
                f"dimension {self.dimension}."
            )
        if not np.isfinite(query).all():
            raise InvalidVectorRecordError(
                "Query embedding contains non-finite values."
            )

        if k == 0 or self.index.ntotal == 0:
            return []

        result_count = min(k, self.index.ntotal)
        distances, positions = self.index.search(query.reshape(1, -1), result_count)
        results: list[dict[str, Any]] = []
        for distance, position in zip(distances[0], positions[0], strict=True):
            if position < 0 or position >= len(self._records):
                raise CorruptSnapshotError(
                    f"FAISS returned invalid record position {position}."
                )
            record = copy.deepcopy(self._records[position])
            record["distance"] = float(distance)
            results.append(record)
        return results

    def get_record(self, chunk_id: str) -> dict[str, Any] | None:
        """Return a defensive copy of one record by chunk identifier.

        Parameters
        ----------
        chunk_id
            Exact canonical chunk identifier.

        Returns
        -------
        dict or None
            Matching record copy, or ``None`` when the identifier is unknown.
        """

        record = self._records_by_id.get(chunk_id)
        return copy.deepcopy(record) if record is not None else None

    def save_snapshot(self) -> None:
        """Persist current memory as one complete atomically selected generation.

        Raises
        ------
        FAISSStoreError
            If the store has no snapshot directory or publication fails.
        CorruptSnapshotError
            If active index and record state are inconsistent.
        """

        if self.snapshot_directory is None:
            raise FAISSStoreError(
                "Cannot save an in-memory store without a snapshot_directory."
            )
        self._validate_index_and_records(self.index, self._records)
        self._write_atomic_snapshot(self.index, self._records)

    def load_snapshot(self) -> None:
        """Load and validate the generation referenced by ``CURRENT``.

        Raises
        ------
        SnapshotNotFoundError
            If no snapshot directory or current pointer exists.
        CorruptSnapshotError
            If the generation is incomplete or internally inconsistent.
        IncompatibleSnapshotError
            If schema version or embedding model differs from configuration.
        DimensionMismatchError
            If persisted index dimension differs from configuration.

        Notes
        -----
        Active memory changes only after the complete generation is validated.
        """

        if self.snapshot_directory is None:
            raise SnapshotNotFoundError(
                "Cannot load a snapshot without a snapshot_directory."
            )

        current_path = self.snapshot_directory / CURRENT_FILENAME
        try:
            generation_name = current_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise SnapshotNotFoundError(
                f"No snapshot pointer found at {current_path}."
            ) from exc
        except OSError as exc:
            raise CorruptSnapshotError(
                f"Could not read snapshot pointer {current_path}: {exc}"
            ) from exc

        if (
            not generation_name.startswith("snapshot-")
            or Path(generation_name).name != generation_name
        ):
            raise CorruptSnapshotError(
                f"Invalid snapshot generation name: {generation_name!r}."
            )

        generation_directory = self.snapshot_directory / "snapshots" / generation_name
        index, records, manifest = self._read_generation(generation_directory)

        schema_version = manifest.get("schema_version")
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise IncompatibleSnapshotError(
                f"Unsupported snapshot schema {schema_version!r}; expected "
                f"{SNAPSHOT_SCHEMA_VERSION}."
            )
        snapshot_dimension = manifest.get("embedding_dimension")
        if snapshot_dimension != self.dimension:
            raise DimensionMismatchError(
                f"Snapshot dimension {snapshot_dimension!r} does not match configured "
                f"dimension {self.dimension}."
            )
        snapshot_model = manifest.get("embedding_model")
        if snapshot_model != self.embedding_model:
            raise IncompatibleSnapshotError(
                f"Snapshot embedding model {snapshot_model!r} does not match "
                f"configured model {self.embedding_model!r}."
            )

        self._validate_index_and_records(index, records)
        self.index = index
        self._set_records(records)

    def _normalise_embedded_chunks(
        self, embedded_chunks: Iterable[Mapping[str, Any]]
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        vectors: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for position, chunk in enumerate(embedded_chunks):
            if not isinstance(chunk, Mapping):
                raise InvalidVectorRecordError(
                    f"Embedded chunk at position {position} must be a mapping."
                )
            try:
                ingestion.chunker.PDFChunker.validate_chunk(chunk)
            except ingestion.chunker.InvalidChunkError as exc:
                raise InvalidVectorRecordError(
                    f"Embedded chunk at position {position} violates the shared "
                    f"chunk schema: {exc}"
                ) from exc

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            metadata = chunk.get("metadata")
            embedding = chunk.get("embedding")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise InvalidVectorRecordError(
                    f"Embedded chunk at position {position} has an invalid chunk_id."
                )
            if not isinstance(text, str) or not text:
                raise InvalidVectorRecordError(
                    f"Embedded chunk {chunk_id!r} must contain non-empty text."
                )
            if not isinstance(metadata, dict):
                raise InvalidVectorRecordError(
                    f"Embedded chunk {chunk_id!r} metadata must be a dictionary."
                )

            try:
                vector = np.asarray(embedding, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise InvalidVectorRecordError(
                    f"Embedding for chunk {chunk_id!r} must contain numeric values."
                ) from exc
            if vector.ndim != 1 or vector.shape[0] != self.dimension:
                actual = vector.shape[0] if vector.ndim == 1 else tuple(vector.shape)
                raise DimensionMismatchError(
                    f"Embedding for chunk {chunk_id!r} has dimension {actual!r}; "
                    f"expected {self.dimension}."
                )
            if not np.isfinite(vector).all():
                raise InvalidVectorRecordError(
                    f"Embedding for chunk {chunk_id!r} contains non-finite values."
                )

            vectors.append(vector)
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": copy.deepcopy(metadata),
                }
            )

        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32), []
        return np.vstack(vectors).astype(np.float32, copy=False), records

    def _set_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._records = [copy.deepcopy(dict(record)) for record in records]
        self._records_by_id = {record["chunk_id"]: record for record in self._records}

    def _validate_index_and_records(
        self, index: faiss.Index, records: Sequence[Mapping[str, Any]]
    ) -> None:
        if index.d != self.dimension:
            raise DimensionMismatchError(
                f"FAISS index dimension {index.d} does not match configured dimension "
                f"{self.dimension}."
            )
        if index.ntotal != len(records):
            raise CorruptSnapshotError(
                f"FAISS index contains {index.ntotal} vectors but snapshot contains "
                f"{len(records)} records."
            )

        ids: list[str] = []
        for position, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise CorruptSnapshotError(
                    f"Snapshot record at position {position} is not an object."
                )
            chunk_id = record.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise CorruptSnapshotError(
                    f"Snapshot record at position {position} has an invalid chunk_id."
                )
            if not isinstance(record.get("text"), str):
                raise CorruptSnapshotError(
                    f"Snapshot record {chunk_id!r} has invalid text."
                )
            if not isinstance(record.get("metadata"), dict):
                raise CorruptSnapshotError(
                    f"Snapshot record {chunk_id!r} metadata is not a dictionary."
                )
            try:
                ingestion.chunker.PDFChunker.validate_chunk(record)
            except ingestion.chunker.InvalidChunkError as exc:
                raise CorruptSnapshotError(
                    f"Snapshot record {chunk_id!r} violates the shared chunk "
                    f"schema: {exc}"
                ) from exc
            ids.append(chunk_id)

        duplicates = sorted(
            chunk_id for chunk_id, count in Counter(ids).items() if count > 1
        )
        if duplicates:
            raise DuplicateChunkIDError(
                f"Snapshot contains duplicate chunk IDs: {duplicates}"
            )

    def _manifest(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "embedding_dimension": self.dimension,
            "embedding_model": self.embedding_model,
            "index_filename": INDEX_FILENAME,
            "records": [copy.deepcopy(dict(record)) for record in records],
        }

    def _write_atomic_snapshot(
        self, index: faiss.Index, records: Sequence[Mapping[str, Any]]
    ) -> None:
        assert self.snapshot_directory is not None
        root = self.snapshot_directory
        snapshots_directory = root / "snapshots"
        root.mkdir(parents=True, exist_ok=True)
        snapshots_directory.mkdir(exist_ok=True)

        generation_name = f"snapshot-{uuid.uuid4().hex}"
        pending_directory = root / f".pending-{uuid.uuid4().hex}"
        pending_directory.mkdir()
        generation_directory = snapshots_directory / generation_name
        pointer_temporary: Path | None = None

        try:
            index_path = pending_directory / INDEX_FILENAME
            serialised_index = faiss.serialize_index(index)
            with index_path.open("wb") as index_file:
                index_file.write(np.asarray(serialised_index, dtype=np.uint8).tobytes())
                index_file.flush()
                os.fsync(index_file.fileno())
            manifest_path = pending_directory / MANIFEST_FILENAME
            with manifest_path.open("w", encoding="utf-8") as manifest_file:
                json.dump(
                    self._manifest(records),
                    manifest_file,
                    ensure_ascii=False,
                    indent=2,
                )
                manifest_file.flush()
                os.fsync(manifest_file.fileno())

            # Verify exactly what will become visible before moving the generation.
            candidate_index, candidate_records, _ = self._read_generation(
                pending_directory
            )
            self._validate_index_and_records(candidate_index, candidate_records)

            os.replace(pending_directory, generation_directory)

            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".CURRENT-", dir=root, text=True
            )
            pointer_temporary = Path(temporary_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as pointer_file:
                pointer_file.write(generation_name)
                pointer_file.flush()
                os.fsync(pointer_file.fileno())
            os.replace(pointer_temporary, root / CURRENT_FILENAME)
            pointer_temporary = None
        except FAISSStoreError:
            raise
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise FAISSStoreError(f"Could not save FAISS snapshot: {exc}") from exc
        finally:
            if pending_directory.exists():
                shutil.rmtree(pending_directory, ignore_errors=True)
            if pointer_temporary is not None and pointer_temporary.exists():
                pointer_temporary.unlink(missing_ok=True)

    def _read_generation(
        self, generation_directory: Path
    ) -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]]:
        index_path = generation_directory / INDEX_FILENAME
        manifest_path = generation_directory / MANIFEST_FILENAME
        if not index_path.is_file() or not manifest_path.is_file():
            raise CorruptSnapshotError(
                f"Snapshot generation {generation_directory} is incomplete; both "
                f"{INDEX_FILENAME} and {MANIFEST_FILENAME} are required."
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            serialised_index = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
            if not serialised_index.size:
                raise ValueError("FAISS index file is empty")
            index = faiss.deserialize_index(serialised_index.copy())
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            raise CorruptSnapshotError(
                f"Could not read snapshot generation {generation_directory}: {exc}"
            ) from exc

        if not isinstance(manifest, dict):
            raise CorruptSnapshotError("Snapshot manifest must be a JSON object.")
        if manifest.get("index_filename") != INDEX_FILENAME:
            raise CorruptSnapshotError(
                f"Snapshot index filename must be {INDEX_FILENAME!r}."
            )
        records = manifest.get("records")
        if not isinstance(records, list):
            raise CorruptSnapshotError("Snapshot records must be a JSON array.")
        return index, records, manifest
