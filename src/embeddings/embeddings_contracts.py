"""
===============================================================================
embeddings_contracts.py
===============================================================================
Define provider-neutral embedding interfaces and failures.

Responsibilities:
  - Specify separate document and query embedding operations.
  - Define stable model and dimension metadata for compatibility checks.
  - Define the project-owned embedding failure boundary.

Design principles:
  - Expose only behavior required by ingestion and retrieval.

Boundaries:
  - Contains no model construction or vector-store behavior.
  - Does not prescribe a specific embedding SDK or execution service.
===============================================================================
"""

from __future__ import annotations

from typing import Protocol, Sequence

__all__ = ["EmbeddingError", "EmbeddingProvider"]


class EmbeddingError(RuntimeError):
    """Represent a UI-safe embedding load, execution, or validation failure."""


class EmbeddingProvider(Protocol):
    """Abstract the embedding capability shared by ingestion and retrieval.

    Implementations must expose a stable model identifier and fixed dimension,
    preserve input order for document vectors, and reject unusable output with
    :class:`EmbeddingError`. Importing an implementation must not load a model.
    """

    @property
    def model_id(self) -> str:
        """Return the provider's stable model identifier."""

        ...

    @property
    def dimension(self) -> int:
        """Return the fixed output-vector dimension."""

        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document passages while preserving input order.

        Parameters
        ----------
        texts
            Ordered non-empty document passages.

        Returns
        -------
        list of list of float
            One fixed-dimension vector per input passage in matching order.
        """

        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query with provider-specific query semantics.

        Parameters
        ----------
        text
            Non-empty query text.

        Returns
        -------
        list of float
            One fixed-dimension query vector compatible with document vectors.
        """

        ...
