"""
===============================================================================
embeddings_sentence_transformer.py
===============================================================================
Run validated local text embeddings with SentenceTransformers.

Responsibilities:
  - Load one injected or local SentenceTransformer instance on first use.
  - Apply retrieval-specific prefixes and normalized batched encoding.
  - Reject invalid, non-finite, or dimensionally incompatible vectors.

Design principles:
  - Load lazily, batch passages, and keep query semantics explicit.
  - Normalize vectors at the model boundary for consistent FAISS search.

Boundaries:
  - Performs no model loading at import time.
  - Does not index vectors or select generation providers.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from . import embeddings_contracts as contracts

__all__ = ["SentenceTransformerEmbeddingProvider"]


class SentenceTransformerEmbeddingProvider:
    """Provide lazy normalized local embeddings through SentenceTransformers.

    Parameters
    ----------
    model_id
        Stable model identifier persisted with vector-store snapshots.
    dimension
        Positive output dimension required from every encoded vector.
    batch_size
        Positive number of document passages encoded per model batch.
    use_e5_prefixes
        Whether to apply ``passage:`` and ``query:`` retrieval prefixes.
    model_factory
        Optional factory used to construct the model on first embedding call.

    Notes
    -----
    The provider caches one model instance after first use. Encoded vectors are
    normalized by the model call and validated for shape and finite values.
    """

    def __init__(
        self,
        *,
        model_id: str,
        dimension: int,
        batch_size: int = 32,
        use_e5_prefixes: bool = True,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """Configure lazy model loading, vector shape, batching, and prefixes."""

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError("dimension must be a positive integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        self._model_id = model_id.strip()
        self._dimension = dimension
        self._batch_size = batch_size
        self._use_e5_prefixes = use_e5_prefixes
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        """Return the persisted model identifier."""

        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""

        return self._dimension

    @staticmethod
    def _default_model_factory(model_id: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_id)

    def _loaded_model(self) -> Any:
        if self._model is None:
            try:
                model = self._model_factory(self.model_id)
                reported_dimension = model.get_sentence_embedding_dimension()
            except Exception as exc:
                raise contracts.EmbeddingError(
                    "The local embedding model could not be loaded."
                ) from exc
            if (
                reported_dimension is not None
                and int(reported_dimension) != self.dimension
            ):
                raise contracts.EmbeddingError(
                    "The configured embedding dimension does not match the local model."
                )
            self._model = model
        return self._model

    def _encode(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every embedding input must be a non-empty string.")

        prepared = [f"{prefix}{text.strip()}" for text in texts]
        try:
            encoded = self._loaded_model().encode(
                prepared,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except contracts.EmbeddingError:
            raise
        except Exception as exc:
            raise contracts.EmbeddingError(
                "The local embedding model could not encode the supplied text."
            ) from exc

        array = np.asarray(encoded, dtype=np.float32)
        expected_shape = (len(prepared), self.dimension)
        if array.shape != expected_shape:
            raise contracts.EmbeddingError(
                "The local embedding model returned an incompatible vector shape."
            )
        if not np.isfinite(array).all():
            raise contracts.EmbeddingError(
                "The local embedding model returned non-finite vector values."
            )
        return array.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ordered passages with configured document-prefix semantics.

        Parameters
        ----------
        texts
            Ordered non-empty passages to encode in configured batches.

        Returns
        -------
        list of list of float
            Normalized vectors in the same order as ``texts``.

        Raises
        ------
        ValueError
            If an input passage is empty or has an invalid type.
        contracts.EmbeddingError
            If model loading, encoding, or output validation fails.
        """

        prefix = "passage: " if self._use_e5_prefixes else ""
        return self._encode(texts, prefix=prefix)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with configured query-prefix semantics.

        Parameters
        ----------
        text
            Non-empty retrieval query.

        Returns
        -------
        list of float
            Normalized fixed-dimension query vector.

        Raises
        ------
        ValueError
            If the query is empty or has an invalid type.
        contracts.EmbeddingError
            If model loading, encoding, or output validation fails.
        """

        prefix = "query: " if self._use_e5_prefixes else ""
        return self._encode([text], prefix=prefix)[0]
