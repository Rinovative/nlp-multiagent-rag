"""
===============================================================================
app.py
===============================================================================
Run the Streamlit boundary for the multilingual PDF RAG application.

Responsibilities:
  - Resolve Streamlit secrets and environment-backed configuration.
  - Maintain one isolated application session across Streamlit reruns.
  - Render PDF upload, chat, provider attribution, and safe errors.

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


def _streamlit_secrets() -> dict[str, object]:
    """Translate optional Streamlit secrets to the canonical flat mapping."""

    try:
        return dict(st.secrets)
    except StreamlitSecretNotFoundError:
        return {}


load_dotenv()
try:
    config = configuration.runtime.AppConfig.from_sources(secrets=_streamlit_secrets())
except configuration.runtime.ConfigurationError as exc:
    st.error(str(exc))
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

st.title("Multilingual PDF RAG")
st.write("Upload one or more PDF documents, then ask questions in German or English.")
if config.huggingface_api_token is None:
    huggingface_is_primary = config.generation_provider == "huggingface" or (
        config.generation_provider == "auto" and not config.openai_is_configured
    )
    if huggingface_is_primary:
        st.info(
            "PDF indexing is local. The selected answer route requires "
            "HUGGINGFACE_API_TOKEN only when a question is submitted."
        )
    elif config.generation_provider == "auto" or config.openai_fallback_enabled:
        st.info(
            "PDF indexing is local. The Hugging Face fallback requires "
            "HUGGINGFACE_API_TOKEN only if that route is invoked."
        )

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

try:
    uploads = [
        application.session.UploadedDocument(
            file_name=uploaded_file.name,
            content=uploaded_file.getvalue(),
        )
        for uploaded_file in uploaded_files
    ]
    with st.spinner("Indexing documents locally..."):
        sync_result = application_session.sync_uploads(uploads)
except (
    configuration.runtime.ConfigurationError,
    embeddings.contracts.EmbeddingError,
    ingestion.processor.DocumentProcessingError,
    vectorstore.faiss.FAISSStoreError,
    application.session.UploadValidationError,
) as exc:
    st.error(str(exc))
else:
    if sync_result.changed and sync_result.processed:
        chunk_count = sum(result.chunk_count for result in sync_result.processed)
        st.success(
            f"Indexed {len(sync_result.processed)} document(s) as "
            f"{chunk_count} retrievable chunks."
        )
    elif sync_result.changed and not uploads:
        st.info("The temporary documents for this session were removed.")

with st.form("question_form"):
    user_query = st.text_input("Ask a question about the uploaded documents")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not user_query.strip():
        st.warning("Enter a question first.")
    elif application_session.vector_store.record_count == 0:
        st.warning("Upload and index at least one PDF first.")
    else:
        try:
            with st.spinner("Retrieving context and generating an answer..."):
                result = application_session.ask(user_query)
        except (
            agents.retriever.RetrievalError,
            configuration.runtime.ConfigurationError,
            embeddings.contracts.EmbeddingError,
            providers.contracts.GenerationError,
            quota.contracts.QuotaError,
            vectorstore.faiss.FAISSStoreError,
        ) as exc:
            st.error(str(exc))
        else:
            st.write(result.answer)
            st.caption(f"Answer provider: {result.provider_id} ({result.model_id})")
            if result.fallback_occurred:
                st.info(
                    "The Hugging Face provider answered because OpenAI capacity "
                    "was unavailable."
                )
