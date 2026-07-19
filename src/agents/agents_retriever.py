"""
===============================================================================
agents_retriever.py
===============================================================================
Retrieve relevant records from one compatible vector store.

Responsibilities:
  - Bound and serialize the current question with recent user history.
  - Validate embedding-model and vector-dimension compatibility.
  - Embed the query and return the nearest session-owned records.

Design principles:
  - Place the current question first so truncation preserves intent.
  - Reject incompatible stores before attempting a search.

Boundaries:
  - Does not mutate the vector store or construct embedding models.
  - Does not generate answers or persist retrieval results.
===============================================================================
"""

from __future__ import annotations

from typing import Any, Sequence

from src import embeddings, vectorstore

__all__ = [
    "RetrievalConfigurationError",
    "RetrievalError",
    "RetrievalValidationError",
    "RetrieverAgent",
]


class RetrievalError(RuntimeError):
    """Represent a project-owned retrieval failure safe for the UI boundary."""


class RetrievalConfigurationError(RetrievalError):
    """Indicate that query embeddings cannot match the active FAISS store."""


class RetrievalValidationError(RetrievalError):
    """Indicate that an invalid question was rejected before embedding."""


class RetrieverAgent:
    """Embed history-aware questions against one session-owned FAISS store.

    Parameters
    ----------
    faiss_store
        Active session store whose model and dimension must match the embedder.
    embedder
        Provider shared with document ingestion for compatible query vectors.
    top_k
        Maximum number of nearest records returned per question.
    max_query_characters
        Character bound for the current question and retained user history.

    Raises
    ------
    ValueError
        If either numeric bound is not a positive integer.
    RetrievalConfigurationError
        If the store and embedder use different models or dimensions.
    """

    def __init__(
        self,
        faiss_store: vectorstore.faiss.FAISSStore,
        embedder: embeddings.contracts.EmbeddingProvider,
        *,
        top_k: int = 5,
        max_query_characters: int = 4_000,
    ) -> None:
        """Create a retriever sharing the ingestion embedding provider."""

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not isinstance(max_query_characters, int) or max_query_characters <= 0:
            raise ValueError("max_query_characters must be a positive integer")
        if faiss_store.embedding_model != embedder.model_id:
            raise RetrievalConfigurationError(
                "The query embedding model does not match the active vector store."
            )
        if faiss_store.dimension != embedder.dimension:
            raise RetrievalConfigurationError(
                "The query embedding dimension does not match the active vector store."
            )
        self.faiss_store = faiss_store
        self.embedder = embedder
        self.top_k = top_k
        self.max_query_characters = max_query_characters

    def retrieve_documents(
        self, query: str, history: Sequence[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """Embed one history-aware query and return nearest records.

        Parameters
        ----------
        query
            Non-empty current question placed before retained history.
        history
            Prior messages; only recent user questions are used for retrieval.

        Returns
        -------
        list of dict
            Up to ``top_k`` nearest records in ascending FAISS distance order.

        Raises
        ------
        RetrievalValidationError
            If the current question is empty or exceeds the configured bound.
        embeddings.contracts.EmbeddingError
            If query embedding fails or returns an invalid vector.
        """

        if not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("The question must not be empty.")
        question = query.strip()
        current_block = f"Current question:\n{question}"
        if len(current_block) > self.max_query_characters:
            raise RetrievalValidationError(
                "The question exceeds the retrieval input limit."
            )
        if self.faiss_store.record_count == 0:
            return []

        history_prefix = "\n\nRecent previous questions:\n"
        remaining = self.max_query_characters - len(current_block) - len(history_prefix)
        prior_questions: list[str] = []
        for message in reversed(history):
            content = message.get("content")
            if (
                message.get("role") != "user"
                or not isinstance(content, str)
                or not content.strip()
            ):
                continue
            if remaining <= 0:
                break
            bounded = content.strip()[:remaining]
            prior_questions.append(bounded)
            remaining -= len(bounded) + 1
        embedding_input = current_block
        if prior_questions:
            prior_questions.reverse()
            embedding_input += history_prefix + "\n".join(prior_questions)
        query_embedding = self.embedder.embed_query(embedding_input)
        return self.faiss_store.search(query_embedding, k=self.top_k)
