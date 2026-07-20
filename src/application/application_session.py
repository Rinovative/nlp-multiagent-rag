"""
===============================================================================
application_session.py
===============================================================================
Own uploads, vector state, and chat lifecycle for one application session.

Responsibilities:
  - Validate and atomically activate one browser session's upload set.
  - Reuse unchanged prepared documents and rebuild the graph when needed.
  - Expose the session-owned question-answering boundary.

Design principles:
  - Commit candidate upload state only after complete successful preparation.
  - Identify uploaded content with deterministic SHA-256 digests.

Boundaries:
  - Does not parse PDFs, create provider clients, or render Streamlit elements.
  - Does not persist session state beyond the owning application lifecycle.
===============================================================================
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from src import ingestion, memory, orchestration, vectorstore

__all__ = [
    "ApplicationSession",
    "SessionDocumentManager",
    "UploadedDocument",
    "UploadSyncResult",
    "UploadValidationError",
]


class UploadValidationError(ValueError):
    """Represent a UI-safe upload rejection before document processing."""


@dataclass(frozen=True)
class UploadedDocument:
    """Represent one immutable in-memory upload owned by a browser session.

    Parameters
    ----------
    file_name
        Non-empty original filename used for source attribution.
    content
        Non-empty PDF bytes retained only by the owning session.

    Raises
    ------
    UploadValidationError
        If the filename or byte content is empty or invalid.

    Notes
    -----
    The value is frozen. Its SHA-256 content hash, not its filename, identifies
    the document throughout ingestion and upload synchronization.
    """

    file_name: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.file_name, str) or not self.file_name.strip():
            raise UploadValidationError("Every upload needs a file name.")
        if not isinstance(self.content, bytes) or not self.content:
            raise UploadValidationError(
                f"The uploaded file {self.file_name!r} is empty or unreadable."
            )

    @property
    def content_hash(self) -> str:
        """Return the content-derived document identity."""

        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True)
class UploadSyncResult:
    """Describe an immutable upload-set synchronization outcome.

    Parameters
    ----------
    changed
        Whether the active content-hash signature changed.
    processed
        Per-document processing summaries in active upload order.
    active_document_ids
        SHA-256 identifiers of the active deduplicated documents.
    """

    changed: bool
    processed: tuple[ingestion.processor.ProcessingResult, ...]
    active_document_ids: tuple[str, ...]


class SessionDocumentManager:
    """Atomically manage the active uploads for one application session.

    Parameters
    ----------
    store_factory
        Factory for isolated empty FAISS stores.
    processor_factory
        Factory that binds ingestion to a candidate store.
    max_upload_file_bytes
        Positive byte bound applied to every selected PDF.
    max_upload_total_bytes
        Positive byte bound applied to the complete selected upload set.
    max_upload_files
        Positive maximum number of selected PDFs.

    Notes
    -----
    A changed upload set is prepared in a candidate store. Active state changes
    only after every new document has been prepared and indexed successfully.
    """

    def __init__(
        self,
        *,
        store_factory: Callable[[], vectorstore.faiss.FAISSStore],
        processor_factory: Callable[
            [vectorstore.faiss.FAISSStore], ingestion.processor.DocumentProcessor
        ],
        max_upload_file_bytes: int,
        max_upload_total_bytes: int,
        max_upload_files: int,
    ) -> None:
        """Create a manager with candidate-store factories and upload bounds."""

        for name, value in (
            ("max_upload_file_bytes", max_upload_file_bytes),
            ("max_upload_total_bytes", max_upload_total_bytes),
            ("max_upload_files", max_upload_files),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_upload_total_bytes < max_upload_file_bytes:
            raise ValueError(
                "max_upload_total_bytes must be at least max_upload_file_bytes"
            )
        self._store_factory = store_factory
        self._processor_factory = processor_factory
        self.max_upload_file_bytes = max_upload_file_bytes
        self.max_upload_total_bytes = max_upload_total_bytes
        self.max_upload_files = max_upload_files
        self.store = store_factory()
        self._active_signature: tuple[str, ...] = ()
        self._prepared_by_hash: dict[str, ingestion.processor.PreparedDocument] = {}

    def sync(self, uploads: Sequence[UploadedDocument]) -> UploadSyncResult:
        """Atomically activate a deduplicated and bounded upload set.

        Parameters
        ----------
        uploads
            Session-owned in-memory documents in the requested active order.

        Returns
        -------
        UploadSyncResult
            Change flag, processing summaries, and active SHA-256 identities.

        Raises
        ------
        UploadValidationError
            If the file-count, individual-size, or combined-size limit is exceeded.
        RuntimeError
            If candidate preparation, embedding, indexing, or persistence fails.

        Notes
        -----
        Unchanged prepared documents are reused by content hash. Any failure
        leaves the previous active store and upload signature intact.
        """

        if len(uploads) > self.max_upload_files:
            raise UploadValidationError(
                f"Select no more than {self.max_upload_files} PDF files at once."
            )

        for upload in uploads:
            if len(upload.content) > self.max_upload_file_bytes:
                raise UploadValidationError(
                    f"The file {upload.file_name!r} exceeds the per-PDF limit of "
                    f"{self.max_upload_file_bytes} bytes."
                )

        total_upload_bytes = sum(len(upload.content) for upload in uploads)
        if total_upload_bytes > self.max_upload_total_bytes:
            raise UploadValidationError(
                "The selected PDFs exceed the combined limit of "
                f"{self.max_upload_total_bytes} bytes."
            )

        unique_uploads: list[UploadedDocument] = []
        seen_hashes: set[str] = set()
        for upload in uploads:
            if upload.content_hash not in seen_hashes:
                unique_uploads.append(upload)
                seen_hashes.add(upload.content_hash)

        signature = tuple(sorted(seen_hashes))
        if signature == self._active_signature:
            return UploadSyncResult(
                changed=False,
                processed=(),
                active_document_ids=self._active_signature,
            )

        candidate_store = self._store_factory()
        processor: ingestion.processor.DocumentProcessor | None = None
        candidate_prepared: dict[str, ingestion.processor.PreparedDocument] = {}
        newly_processed: list[ingestion.processor.ProcessingResult] = []

        for upload in unique_uploads:
            prepared = self._prepared_by_hash.get(upload.content_hash)
            if prepared is None:
                if processor is None:
                    processor = self._processor_factory(candidate_store)
                prepared = processor.prepare_bytes(
                    upload.content, file_name=upload.file_name
                )
                newly_processed.append(prepared.result)
            candidate_prepared[upload.content_hash] = prepared
            candidate_store.add_embedded_chunks(prepared.embedded_chunks)

        # Commit only after every active document has been prepared and indexed.
        self.store = candidate_store
        self._prepared_by_hash = candidate_prepared
        self._active_signature = signature
        return UploadSyncResult(
            changed=True,
            processed=tuple(newly_processed),
            active_document_ids=signature,
        )

    @property
    def active_document_count(self) -> int:
        """Return the number of content-unique active documents."""

        return len(self._active_signature)

    def clear(self) -> None:
        """Discard all documents and cached preparations for this manager.

        Notes
        -----
        The active store is replaced with a fresh isolated empty store.
        """

        self.store = self._store_factory()
        self._active_signature = ()
        self._prepared_by_hash = {}


class ApplicationSession:
    """Own all mutable application state for one browser session.

    Parameters
    ----------
    session_id
        Non-empty identifier used for conversation and quota isolation.
    document_manager
        Manager for this session's active documents and vector store.
    conversation_store
        Store containing only explicitly keyed conversation histories.
    chatbot_factory
        Factory that binds a graph to the current store and conversation memory.

    Notes
    -----
    The graph is rebuilt lazily whenever the active upload set changes. State is
    process-local unless an injected dependency explicitly persists it.
    """

    def __init__(
        self,
        *,
        session_id: str,
        document_manager: SessionDocumentManager,
        conversation_store: memory.contracts.ConversationStore,
        chatbot_factory: Callable[
            [vectorstore.faiss.FAISSStore, memory.contracts.ConversationStore],
            orchestration.rag.RAGChatbot,
        ],
    ) -> None:
        """Create an application session from isolated mutable resources."""

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        self.session_id = session_id
        self.document_manager = document_manager
        self.conversation_store = conversation_store
        self._chatbot_factory = chatbot_factory
        self._chatbot: orchestration.rag.RAGChatbot | None = None

    @property
    def vector_store(self) -> vectorstore.faiss.FAISSStore:
        """Return the currently active session-owned vector store."""

        return self.document_manager.store

    @property
    def active_document_count(self) -> int:
        """Return the number of active content-unique documents."""

        return self.document_manager.active_document_count

    def sync_uploads(self, uploads: Sequence[UploadedDocument]) -> UploadSyncResult:
        """Activate uploads and invalidate the graph when documents change.

        Parameters
        ----------
        uploads
            Complete requested upload set for this session.

        Returns
        -------
        UploadSyncResult
            Atomic synchronization outcome from the document manager.
        """

        result = self.document_manager.sync(uploads)
        if result.changed:
            self._chatbot = None
        return result

    def ask(self, question: str) -> orchestration.rag.RAGResult:
        """Answer one question through the lazily rebuilt session graph.

        Parameters
        ----------
        question
            Non-empty user question for this session's active documents.

        Returns
        -------
        orchestration.rag.RAGResult
            Answer with actual provider attribution and ranked source references.

        Raises
        ------
        ValueError
            If the orchestration boundary rejects an empty question.
        providers.contracts.GenerationError
            If provider configuration, execution, or response validation fails.
        RuntimeError
            If another project-owned retrieval, embedding, storage, or quota
            boundary fails.

        Notes
        -----
        Retrieval, embedding, and quota errors propagate through this boundary so
        the Streamlit adapter can render their project-owned safe messages.
        """

        if self._chatbot is None:
            self._chatbot = self._chatbot_factory(
                self.document_manager.store, self.conversation_store
            )
        return self._chatbot.process_user_input(question, chat_id=self.session_id)

    def close(self) -> None:
        """Release this session's documents, history, and compiled graph.

        Notes
        -----
        Closing affects only the explicitly owned session resources.
        """

        self.document_manager.clear()
        self.conversation_store.clear(self.session_id)
        self._chatbot = None
