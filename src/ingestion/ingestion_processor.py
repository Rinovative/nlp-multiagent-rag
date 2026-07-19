"""
===============================================================================
ingestion_processor.py
===============================================================================
Coordinate PDF ingestion from uploaded bytes without shared files.

Responsibilities:
  - Run loading, preprocessing, chunking, embedding, and optional indexing.
  - Verify content-derived document identities across the pipeline.
  - Translate unexpected parser failures into UI-safe project errors.

Design principles:
  - Prepare immutable results before mutating a target vector store.
  - Preserve project-owned validation errors without leaking SDK details.

Boundaries:
  - Delegates each transformation to its focused domain component.
  - Does not own upload-set activation or session lifecycle.
===============================================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from src import embeddings, vectorstore

from . import ingestion_chunker as chunker
from . import ingestion_loader as loader_module
from . import ingestion_preprocessing as preprocessing

__all__ = [
    "DocumentProcessingError",
    "DocumentProcessor",
    "PreparedDocument",
    "ProcessingResult",
]


class DocumentProcessingError(RuntimeError):
    """Represent a document-processing failure safe for the UI boundary."""


@dataclass(frozen=True)
class ProcessingResult:
    """Summarize one successfully prepared immutable document result.

    Parameters
    ----------
    document_id
        SHA-256 identity derived from the uploaded bytes.
    file_name
        Original filename retained for source attribution.
    chunk_count
        Number of canonical embedded chunks produced.
    """

    document_id: str
    file_name: str
    chunk_count: int


@dataclass(frozen=True)
class PreparedDocument:
    """Hold an immutable prepared result and its canonical embedded chunks.

    Parameters
    ----------
    result
        Public processing summary for the document.
    embedded_chunks
        Ordered tuple ready for insertion into a compatible vector store.
    """

    result: ProcessingResult
    embedded_chunks: tuple[dict[str, Any], ...]


class DocumentProcessor:
    """Load, preprocess, chunk, embed, and optionally index uploaded PDFs.

    Parameters
    ----------
    faiss_store
        Target store mutated only by ``process_bytes`` or ``process_upload``.
    embedding_provider
        Provider used to embed canonical chunks.
    loader
        Optional PDF loader; a fresh ``UniversalPDFLoader`` is used by default.
    chunker_instance
        Optional deterministic chunker.
    preprocessor_factory
        Factory binding one loader mapping to structural preprocessing.
    """

    def __init__(
        self,
        *,
        faiss_store: vectorstore.faiss.FAISSStore,
        embedding_provider: embeddings.contracts.EmbeddingProvider,
        loader: loader_module.UniversalPDFLoader | None = None,
        chunker_instance: chunker.PDFChunker | None = None,
        preprocessor_factory: Callable[[dict], Any] = preprocessing.PdfPreprocessor,
    ) -> None:
        """Create the ingestion coordinator from injectable domain components."""

        self.loader = loader or loader_module.UniversalPDFLoader()
        self.chunker = chunker_instance or chunker.PDFChunker()
        self.embedding_provider = embedding_provider
        self.faiss_store = faiss_store
        self.preprocessor_factory = preprocessor_factory

    def process_upload(self, uploaded_file: Any) -> ProcessingResult:
        """Process a Streamlit-like upload object directly from memory.

        Parameters
        ----------
        uploaded_file
            Object exposing ``name`` and either ``getvalue`` or ``getbuffer``.

        Returns
        -------
        ProcessingResult
            Summary after prepared chunks are added to the target store.

        Raises
        ------
        DocumentProcessingError
            If the upload interface or PDF processing fails.
        TypeError
            If the upload accessor does not return bytes-like content.
        embeddings.contracts.EmbeddingError
            If local embedding fails or returns invalid vectors.
        """

        file_name = getattr(uploaded_file, "name", "uploaded.pdf")
        if hasattr(uploaded_file, "getvalue"):
            content = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "getbuffer"):
            content = bytes(uploaded_file.getbuffer())
        else:
            raise DocumentProcessingError(
                "The uploaded object does not provide readable PDF bytes."
            )
        return self.process_bytes(bytes(content), file_name=file_name)

    def process_bytes(self, content: bytes, *, file_name: str) -> ProcessingResult:
        """Prepare bytes and immediately add their chunks to the target store.

        Parameters
        ----------
        content
            Non-empty PDF bytes.
        file_name
            Original filename retained in source metadata.

        Returns
        -------
        ProcessingResult
            Prepared document summary after store mutation.

        Raises
        ------
        DocumentProcessingError
            If document preparation fails.
        vectorstore.faiss.FAISSStoreError
            If the prepared chunks cannot be indexed or persisted.
        """

        prepared = self.prepare_bytes(content, file_name=file_name)
        self.faiss_store.add_embedded_chunks(prepared.embedded_chunks)
        return prepared.result

    def prepare_bytes(self, content: bytes, *, file_name: str) -> PreparedDocument:
        """Prepare one document without mutating the target vector store.

        Parameters
        ----------
        content
            Non-empty PDF bytes whose SHA-256 digest becomes the document identity.
        file_name
            Original filename retained in canonical source metadata.

        Returns
        -------
        PreparedDocument
            Immutable processing summary and ordered embedded chunks.

        Raises
        ------
        DocumentProcessingError
            If loading, preprocessing, chunking, or identity checks fail.
        embeddings.contracts.EmbeddingError
            If embedding fails or returns unusable vectors.
        """

        if not isinstance(content, bytes) or not content:
            raise DocumentProcessingError("The uploaded PDF is empty.")
        if not isinstance(file_name, str) or not file_name.strip():
            raise DocumentProcessingError("The uploaded PDF needs a file name.")

        document_id = hashlib.sha256(content).hexdigest()
        try:
            document = self.loader.load_pdf(
                content, file_name=file_name, extract_tables=True
            )
            processed_document, _removed = self.preprocessor_factory(
                document
            ).run_preprocessing()
            chunks = self.chunker.chunk_document(processed_document)
            if not chunks:
                raise DocumentProcessingError(
                    "The PDF did not contain any indexable text or tables."
                )
            if any(chunk["metadata"]["document_id"] != document_id for chunk in chunks):
                raise DocumentProcessingError(
                    "The loader and chunker produced inconsistent document IDs."
                )
            embedded_chunks = embeddings.chunks.embed_chunks(
                chunks, self.embedding_provider
            )
        except DocumentProcessingError:
            raise
        except embeddings.contracts.EmbeddingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(
                f"Could not process {file_name!r} as a PDF."
            ) from exc

        return PreparedDocument(
            result=ProcessingResult(
                document_id=document_id,
                file_name=file_name,
                chunk_count=len(chunks),
            ),
            embedded_chunks=tuple(embedded_chunks),
        )
