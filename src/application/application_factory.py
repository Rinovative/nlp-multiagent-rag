"""
===============================================================================
application_factory.py
===============================================================================
Compose session-owned application components from validated configuration.

Responsibilities:
  - Share one lazy local embedding provider across Streamlit reruns.
  - Construct hosted-provider clients only when generation is invoked.
  - Wire session isolation, orchestration, routing, and quota enforcement.

Design principles:
  - Cache immutable expensive resources and inject narrow dependencies.
  - Keep optional hosted services lazy and independently configurable.

Boundaries:
  - Importing this module creates no clients, models, directories, or connections.
  - Does not own Streamlit state or implement domain operations.
===============================================================================
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src import (
    agents,
    configuration,
    embeddings,
    ingestion,
    memory,
    orchestration,
    providers,
    quota,
    vectorstore,
)

from . import application_session as session

__all__ = ["create_application_session", "create_embedding_provider"]


@lru_cache(maxsize=8)
def _cached_embedding_provider(
    model_id: str,
    dimension: int,
    batch_size: int,
    use_e5_prefixes: bool,
) -> embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider:
    return embeddings.sentence_transformer.SentenceTransformerEmbeddingProvider(
        model_id=model_id,
        dimension=dimension,
        batch_size=batch_size,
        use_e5_prefixes=use_e5_prefixes,
    )


def create_embedding_provider(
    config: configuration.runtime.AppConfig,
) -> embeddings.contracts.EmbeddingProvider:
    """Return the process-cached local embedding provider for a configuration.

    Parameters
    ----------
    config
        Validated model identifier, dimension, batch size, and prefix settings.

    Returns
    -------
    embeddings.contracts.EmbeddingProvider
        Lazy provider shared by sessions with the same immutable settings.

    Notes
    -----
    This function caches provider objects, not eagerly loaded model instances.
    """

    return _cached_embedding_provider(
        config.embedding_model,
        config.embedding_dimension,
        config.embedding_batch_size,
        config.embedding_uses_e5_prefixes,
    )


def _generation_router(
    config: configuration.runtime.AppConfig,
) -> providers.router.GenerationRouter:
    hf_clients: list[Any] = []

    def huggingface_client() -> Any:
        if not hf_clients:
            from huggingface_hub import InferenceClient

            hf_clients.append(
                InferenceClient(
                    model=config.huggingface_generation_model,
                    provider="auto",
                    token=config.require_huggingface_token(),
                    timeout=config.provider_timeout_seconds,
                )
            )
        return hf_clients[0]

    free_provider = providers.huggingface.HuggingFaceGenerationProvider(
        huggingface_client,
        model_id=config.huggingface_generation_model,
    )

    openai_provider: providers.contracts.GenerationProvider | None = None
    if config.openai_is_configured:
        openai_clients: list[Any] = []

        def openai_client() -> Any:
            if not openai_clients:
                from openai import OpenAI

                openai_clients.append(
                    OpenAI(
                        api_key=config.require_openai_key(),
                        timeout=config.provider_timeout_seconds,
                        max_retries=0,
                    )
                )
            return openai_clients[0]

        openai_provider = providers.openai.OpenAIGenerationProvider(
            openai_client,
            model_id=config.openai_generation_model,
        )

    quota_backend: quota.contracts.QuotaBackend | None = None
    if config.redis_url is not None:
        quota_backend = quota.redis.RedisQuotaBackend(
            config.redis_url,
            key_prefix=config.quota_key_prefix,
        )

    return providers.router.GenerationRouter(
        mode=config.generation_provider,
        free_provider=free_provider,
        openai_provider=openai_provider,
        quota_backend=quota_backend,
        openai_fallback_enabled=config.openai_fallback_enabled,
    )


def create_application_session(
    *, session_id: str, config: configuration.runtime.AppConfig
) -> session.ApplicationSession:
    """Build one isolated application session from validated configuration.

    Parameters
    ----------
    session_id
        Non-empty browser-session identifier used for memory and quota scope.
    config
        Validated settings used to compose embeddings, providers, and limits.

    Returns
    -------
    session.ApplicationSession
        Session with isolated documents, vector state, memory, and graph lifecycle.

    Raises
    ------
    ValueError
        If ``session_id`` is empty.

    Notes
    -----
    Hosted clients, Redis connections, and the local embedding model remain lazy.
    """

    embedding_provider = create_embedding_provider(config)
    generation_router = _generation_router(config)

    def store_factory() -> vectorstore.faiss.FAISSStore:
        return vectorstore.faiss.FAISSStore(
            dimension=config.embedding_dimension,
            embedding_model=config.embedding_model,
        )

    def processor_factory(
        store: vectorstore.faiss.FAISSStore,
    ) -> ingestion.processor.DocumentProcessor:
        return ingestion.processor.DocumentProcessor(
            faiss_store=store,
            embedding_provider=embedding_provider,
            chunker_instance=ingestion.chunker.PDFChunker(
                max_chunk_length=1000,
                overlap_length=200,
            ),
        )

    conversation_store: memory.contracts.ConversationStore = (
        memory.in_memory.InMemoryConversationStore(
            max_history=config.max_history_messages
        )
    )

    def chatbot_factory(
        store: vectorstore.faiss.FAISSStore,
        session_conversation_store: memory.contracts.ConversationStore,
    ) -> orchestration.rag.RAGChatbot:
        return orchestration.rag.RAGChatbot(
            retriever_agent=agents.retriever.RetrieverAgent(
                store,
                embedding_provider,
                top_k=config.retrieval_top_k,
            ),
            generator_agent=agents.generator.GeneratorAgent(
                generation_router,
                max_input_characters=config.max_input_characters,
                max_output_tokens=config.max_output_tokens,
            ),
            memory_agent=agents.memory.MemoryAgent(session_conversation_store),
        )

    document_manager = session.SessionDocumentManager(
        store_factory=store_factory,
        processor_factory=processor_factory,
        max_upload_bytes=config.max_upload_mb * 1024 * 1024,
    )
    return session.ApplicationSession(
        session_id=session_id,
        document_manager=document_manager,
        conversation_store=conversation_store,
        chatbot_factory=chatbot_factory,
    )
