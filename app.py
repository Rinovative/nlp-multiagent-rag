"""
===============================================================================
app.py
===============================================================================
Run the Streamlit boundary for the multilingual PDF RAG application.

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


st.set_page_config(page_title="Multilingual PDF RAG", layout="wide")


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


def _format_upload_size(size_bytes: int) -> str:
    """Format bytes using the application's documented binary MB unit."""

    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _render_sources(sources: Sequence[Mapping[str, object]]) -> None:
    """Render safe source labels without exposing retrieved chunk text."""

    with st.expander("Verwendete Quellen", expanded=False):
        if not sources:
            st.caption("Für diese Antwort sind keine Seitenangaben verfügbar.")
            return
        for source in sources:
            document_name = str(source.get("document_name", "")).strip()
            if not document_name:
                continue
            page_number = source.get("page_number")
            suffix = (
                f" · Seite {page_number}"
                if isinstance(page_number, int) and page_number > 0
                else ""
            )
            st.text(f"• {document_name}{suffix}")


def _render_chat_entry(entry: Mapping[str, object]) -> None:
    """Render one persisted user or assistant chat entry."""

    role = "user" if entry.get("role") == "user" else "assistant"
    with st.chat_message(role):
        st.write(str(entry.get("content", "")))
        if role == "assistant":
            provider_id = str(entry.get("provider_id", ""))
            model_id = str(entry.get("model_id", ""))
            st.caption(f"Antwortanbieter: {provider_id} ({model_id})")
            if entry.get("fallback_occurred") is True:
                st.info(
                    "Hugging Face hat kontrolliert übernommen, weil der "
                    "OpenAI-Pfad nicht verfügbar war."
                )
            raw_sources = entry.get("sources", [])
            sources = raw_sources if isinstance(raw_sources, list) else []
            _render_sources(sources)


def _safe_ui_error(exc: Exception) -> str:
    """Translate project-owned failures into concise German UI messages."""

    if isinstance(exc, application.session.UploadValidationError):
        return str(exc)
    if isinstance(exc, configuration.runtime.ConfigurationError):
        return "Die Anwendungskonfiguration ist unvollständig oder ungültig."
    if isinstance(exc, agents.retriever.RetrievalError):
        return "Die Frage konnte nicht für die Dokumentensuche verarbeitet werden."
    if isinstance(exc, ingestion.processor.DocumentProcessingError):
        return (
            "Mindestens ein PDF konnte nicht verarbeitet werden. "
            "Das bisherige Upload-Set bleibt aktiv."
        )
    if isinstance(exc, embeddings.contracts.EmbeddingError):
        return "Die Dokumente konnten nicht eingebettet oder durchsucht werden."
    if isinstance(exc, vectorstore.faiss.FAISSStoreError):
        return "Der sitzungseigene Suchindex ist derzeit nicht verfügbar."
    if isinstance(exc, providers.contracts.GenerationSafetyError):
        return "Die Anfrage konnte wegen einer Sicherheitsbeschränkung nicht beantwortet werden."
    if isinstance(exc, providers.contracts.GenerationError):
        return "Die Antwortgenerierung ist derzeit nicht verfügbar."
    if isinstance(exc, quota.contracts.QuotaError):
        return "Das geschützte OpenAI-Kontingent ist derzeit nicht verfügbar."
    return "Die Anfrage konnte nicht verarbeitet werden."


load_dotenv()
try:
    config = configuration.runtime.AppConfig.from_sources(secrets=_streamlit_secrets())
except configuration.runtime.ConfigurationError:
    st.error("Die Anwendungskonfiguration ist unvollständig oder ungültig.")
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
    st.title("Multilingual PDF RAG")
    st.write(
        "Lade PDF-Dokumente hoch. Nach der lokalen Indexierung kannst du im "
        "Hauptbereich Fragen auf Deutsch oder Englisch stellen."
    )
    if config.huggingface_api_token is None:
        huggingface_is_primary = config.generation_provider == "huggingface" or (
            config.generation_provider == "auto" and not config.openai_is_configured
        )
        if huggingface_is_primary:
            st.info(
                "Die PDF-Indexierung erfolgt lokal. Für Antworten über Hugging Face "
                "muss HUGGINGFACE_API_TOKEN konfiguriert sein."
            )
        elif config.generation_provider == "auto" or config.openai_fallback_enabled:
            st.info(
                "Die PDF-Indexierung erfolgt lokal. Ein möglicher Hugging-Face-Fallback "
                "benötigt HUGGINGFACE_API_TOKEN."
            )

    uploaded_files = st.file_uploader(
        "PDF-Dokumente hochladen",
        type=["pdf"],
        accept_multiple_files=True,
        max_upload_size=config.max_upload_file_mb,
        help=(
            f"Höchstens {config.max_upload_files} PDFs, "
            f"{config.max_upload_file_mb} MB pro Datei und "
            f"{config.max_upload_total_mb} MB insgesamt. "
            "Dabei gilt 1 MB = 1 048 576 Byte."
        ),
    )

    uploaded_payloads = [
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in (uploaded_files or [])
    ]
    selected_size = sum(len(content) for _, content in uploaded_payloads)
    st.caption(
        f"Ausgewählt: {len(uploaded_payloads)} von {config.max_upload_files} PDFs, "
        f"zusammen {_format_upload_size(selected_size)} von "
        f"{config.max_upload_total_mb} MB."
    )

    try:
        uploads = [
            application.session.UploadedDocument(
                file_name=file_name,
                content=content,
            )
            for file_name, content in uploaded_payloads
        ]
        with st.spinner("PDF-Dokumente werden lokal verarbeitet und indexiert …"):
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
            chunk_count = sum(result.chunk_count for result in sync_result.processed)
            st.success(
                f"{len(sync_result.processed)} PDF-Dokument(e) wurden erfolgreich "
                f"als {chunk_count} durchsuchbare Chunks indexiert."
            )
        elif sync_result.changed and not uploads:
            st.info("Die temporären Dokumente dieser Sitzung wurden entfernt.")

    st.caption(
        f"Aktiv: {application_session.active_document_count} Dokument(e), "
        f"{application_session.vector_store.record_count} Chunks."
    )

for chat_entry in st.session_state.chat_messages:
    _render_chat_entry(chat_entry)

documents_are_ready = application_session.vector_store.record_count > 0
if not documents_are_ready:
    st.info("Lade zuerst PDF-Dokumente über die Seitenleiste hoch.")

user_query = st.chat_input(
    "Frage zu den indexierten Dokumenten eingeben",
    disabled=not documents_are_ready,
    key="document_question",
)

if user_query:
    user_entry: _ChatEntry = {"role": "user", "content": user_query}
    st.session_state.chat_messages.append(user_entry)
    _render_chat_entry(user_entry)
    try:
        with st.spinner("Relevante Textstellen werden gesucht und beantwortet …"):
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
