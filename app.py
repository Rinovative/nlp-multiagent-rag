"""
===============================================================================
app.py
===============================================================================
Run the Streamlit boundary for the PDF RAG Assistant application.

Responsibilities:
  - Resolve Streamlit secrets and environment-backed configuration.
  - Maintain one isolated application session across Streamlit reruns.
  - Render sidebar document management and a focused conversational main area.
  - Present provider attribution, grounded sources, and safe errors.

Design principles:
  - Keep domain construction lazy and session state explicit.
  - Treat Streamlit as an executable adapter rather than a domain package.

Boundaries:
  - Does not implement ingestion, retrieval, generation, or quota policy.
  - Accesses credentials only at runtime through the Streamlit boundary.
===============================================================================
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from src import (
    agents,
    application,
    configuration,
    embeddings,
    ingestion,
    providers,
    quota,
    vectorstore,
)


st.set_page_config(page_title="PDF RAG Assistant", layout="wide")


def _streamlit_secrets() -> dict[str, object]:
    """Translate optional Streamlit secrets to the canonical flat mapping."""

    try:
        return dict(st.secrets)
    except StreamlitSecretNotFoundError:
        return {}


class _StoredSource(TypedDict):
    document_name: str
    page_number: int | None


class _ChatEntry(TypedDict, total=False):
    role: str
    content: str
    provider_id: str
    model_id: str
    fallback_occurred: bool
    sources: list[_StoredSource]


def _render_sources(sources: Sequence[Mapping[str, object]]) -> None:
    """Render safe source labels without exposing retrieved chunk text."""

    with st.expander("Sources", expanded=False):
        if not sources:
            st.caption("No page references are available for this answer.")
            return
        seen: set[tuple[str, int | None]] = set()
        for source in sources:
            document_name = str(source.get("document_name", "")).strip()
            if not document_name:
                continue
            raw_page_number = source.get("page_number")
            page_number = (
                raw_page_number
                if isinstance(raw_page_number, int) and raw_page_number > 0
                else None
            )
            identity = (document_name, page_number)
            if identity in seen:
                continue
            seen.add(identity)
            suffix = f" · page {page_number}" if page_number is not None else ""
            st.text(f"• {document_name}{suffix}")


def _render_chat_entry(entry: Mapping[str, object]) -> None:
    """Render one persisted user or assistant chat entry."""

    role = "user" if entry.get("role") == "user" else "assistant"
    with st.chat_message(role):
        st.write(str(entry.get("content", "")))
        if role == "assistant":
            provider_id = str(entry.get("provider_id", ""))
            model_id = str(entry.get("model_id", ""))
            provider_name = {
                "huggingface": "Hugging Face",
                "openai": "OpenAI",
            }.get(provider_id, provider_id)
            st.caption(f"{provider_name} · {model_id}")
            raw_sources = entry.get("sources", [])
            sources = raw_sources if isinstance(raw_sources, list) else []
            _render_sources(sources)


def _safe_ui_error(exc: Exception) -> str:
    """Translate project-owned failures into concise English UI messages."""

    if isinstance(exc, application.session.UploadValidationError):
        return str(exc)
    if isinstance(exc, configuration.runtime.ConfigurationError):
        return "Answer generation is not configured. Please contact the site owner."
    if isinstance(exc, agents.retriever.RetrievalError):
        return "The question could not be processed for document search."
    if isinstance(exc, ingestion.processor.DocumentProcessingError):
        return (
            "At least one PDF could not be processed. "
            "The previous document set remains active."
        )
    if isinstance(exc, embeddings.contracts.EmbeddingError):
        return "The documents could not be indexed or searched."
    if isinstance(exc, vectorstore.faiss.FAISSStoreError):
        return "The document search index is currently unavailable."
    if isinstance(exc, providers.contracts.GenerationSafetyError):
        return "The request could not be answered because of a safety restriction."
    if isinstance(
        exc,
        (
            providers.contracts.GenerationConfigurationError,
            providers.contracts.GenerationAuthenticationError,
        ),
    ):
        return "Answer generation is not configured. Please contact the site owner."
    if isinstance(exc, providers.contracts.GenerationFallbackError):
        provider_error_type = exc.provider_error_type
        if issubclass(
            provider_error_type,
            (
                providers.contracts.GenerationAuthenticationError,
                providers.contracts.GenerationConfigurationError,
            ),
        ):
            return (
                "The free fallback provider is not configured correctly. "
                "Please try again later."
            )
        if issubclass(provider_error_type, providers.contracts.GenerationCreditsError):
            return (
                "The free fallback provider has reached its usage limit. "
                "Please try again later."
            )
        if issubclass(
            provider_error_type,
            (
                providers.contracts.GenerationModelUnavailableError,
                providers.contracts.GenerationRateLimitError,
                providers.contracts.GenerationTemporaryError,
            ),
        ):
            return (
                "The free fallback provider is temporarily unavailable. "
                "Please try again later."
            )
        if issubclass(provider_error_type, providers.contracts.GenerationSafetyError):
            return "The request could not be answered because of a safety restriction."
        return "Answer generation is currently unavailable. Please try again later."
    if isinstance(exc, providers.contracts.GenerationModelUnavailableError):
        return "The configured generation model is currently unavailable."
    if isinstance(
        exc,
        (
            providers.contracts.GenerationCreditsError,
            providers.contracts.GenerationRateLimitError,
            providers.contracts.GenerationTemporaryError,
        ),
    ):
        return "Answer generation is temporarily unavailable. Please try again later."
    if isinstance(exc, providers.contracts.GenerationError):
        return "Answer generation is currently unavailable. Please try again later."
    if isinstance(exc, quota.contracts.QuotaError):
        return "The protected OpenAI allowance is currently unavailable."
    return "The request could not be processed."


def _document_status(document_count: int, chunk_count: int) -> str:
    """Format compact active-document state with correct English plurals."""

    document_label = "document" if document_count == 1 else "documents"
    chunk_label = "chunk" if chunk_count == 1 else "chunks"
    return (
        f"{document_count} {document_label} · " f"{chunk_count} {chunk_label} indexed"
    )


load_dotenv()
try:
    config = configuration.runtime.AppConfig.from_sources(secrets=_streamlit_secrets())
except configuration.runtime.ConfigurationError:
    st.error("The application configuration is incomplete or invalid.")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "application_session" not in st.session_state:
    st.session_state.application_session = (
        application.factory.create_application_session(
            session_id=st.session_state.session_id,
            config=config,
        )
    )

application_session = st.session_state.application_session
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.markdown("**PDF RAG Assistant**")
    st.subheader("Documents")
    st.write("Upload PDFs and ask questions about their content")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=config.max_upload_file_mb,
        help=(
            f"Up to {config.max_upload_files} PDFs · "
            f"{config.max_upload_file_mb} MB per file · "
            f"{config.max_upload_total_mb} MB total"
        ),
    )

    uploaded_payloads = [
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in (uploaded_files or [])
    ]
    try:
        uploads = [
            application.session.UploadedDocument(
                file_name=file_name,
                content=content,
            )
            for file_name, content in uploaded_payloads
        ]
        with st.spinner("Processing documents…"):
            sync_result = application_session.sync_uploads(uploads)
    except (
        configuration.runtime.ConfigurationError,
        embeddings.contracts.EmbeddingError,
        ingestion.processor.DocumentProcessingError,
        vectorstore.faiss.FAISSStoreError,
        application.session.UploadValidationError,
    ) as exc:
        st.error(_safe_ui_error(exc))
    else:
        if sync_result.changed and sync_result.processed:
            st.success(
                _document_status(
                    application_session.active_document_count,
                    application_session.vector_store.record_count,
                )
            )
        elif sync_result.changed and not uploads:
            st.info("Documents removed.")
        else:
            st.caption(
                _document_status(
                    application_session.active_document_count,
                    application_session.vector_store.record_count,
                )
            )

for chat_entry in st.session_state.chat_messages:
    _render_chat_entry(chat_entry)

documents_are_ready = application_session.vector_store.record_count > 0
if not documents_are_ready:
    st.caption("Upload PDFs in the sidebar to start chatting.")

user_query = st.chat_input(
    "Ask a question about your documents",
    disabled=not documents_are_ready,
    key="document_question",
)

if user_query:
    user_entry: _ChatEntry = {"role": "user", "content": user_query}
    st.session_state.chat_messages.append(user_entry)
    _render_chat_entry(user_entry)
    try:
        with st.spinner("Generating answer…"):
            result = application_session.ask(user_query)
    except (
        agents.retriever.RetrievalError,
        configuration.runtime.ConfigurationError,
        embeddings.contracts.EmbeddingError,
        providers.contracts.GenerationError,
        quota.contracts.QuotaError,
        vectorstore.faiss.FAISSStoreError,
    ) as exc:
        with st.chat_message("assistant"):
            st.error(_safe_ui_error(exc))
    else:
        assistant_entry: _ChatEntry = {
            "role": "assistant",
            "content": result.answer,
            "provider_id": result.provider_id,
            "model_id": result.model_id,
            "fallback_occurred": result.fallback_occurred,
            "sources": [
                {
                    "document_name": source.document_name,
                    "page_number": source.page_number,
                }
                for source in result.sources
            ],
        }
        st.session_state.chat_messages.append(assistant_entry)
        _render_chat_entry(assistant_entry)
