"""
===============================================================================
embeddings_chunks.py
===============================================================================
Attach provider vectors to canonical PDF chunks.

Responsibilities:
  - Validate the minimal canonical chunk shape.
  - Batch texts through an embedding provider and attach ordered vectors.

Design principles:
  - Preserve caller-owned chunks through defensive metadata copies.
  - Reject provider result counts that cannot preserve chunk ordering.

Boundaries:
  - Delegates vector semantics and dimensional validation to the provider.
  - Does not index or persist the resulting embedded chunks.
===============================================================================
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from . import embeddings_contracts as contracts

__all__ = ["embed_chunks"]


def embed_chunks(
    chunks: Sequence[Mapping[str, Any]], provider: contracts.EmbeddingProvider
) -> list[dict[str, Any]]:
    """Embed canonical chunks without mutating caller-owned values.

    Parameters
    ----------
    chunks
        Ordered mappings containing non-empty ``chunk_id``, ``text``, and
        mapping-valued ``metadata`` fields.
    provider
        Embedding provider used once for the ordered document texts.

    Returns
    -------
    list of dict
        Defensive chunk copies with corresponding ``embedding`` vectors.

    Raises
    ------
    contracts.EmbeddingError
        If a chunk is invalid or the provider returns the wrong vector count.
    """

    validated: list[Mapping[str, Any]] = []
    for chunk in chunks:
        if (
            not isinstance(chunk, Mapping)
            or not isinstance(chunk.get("chunk_id"), str)
            or not chunk["chunk_id"]
            or not isinstance(chunk.get("text"), str)
            or not chunk["text"]
            or not isinstance(chunk.get("metadata"), Mapping)
        ):
            raise contracts.EmbeddingError(
                "A document chunk does not match the canonical embedding schema."
            )
        validated.append(chunk)

    vectors = provider.embed_documents([chunk["text"] for chunk in validated])
    if len(vectors) != len(validated):
        raise contracts.EmbeddingError(
            "The embedding provider returned an unexpected vector count."
        )
    return [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": copy.deepcopy(dict(chunk["metadata"])),
            "embedding": vector,
        }
        for chunk, vector in zip(validated, vectors, strict=True)
    ]
