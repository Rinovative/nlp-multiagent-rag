"""Local embedding contracts and SentenceTransformer implementation.

Provides:
- chunks: deterministic chunk-to-vector mapping.
- contracts: embedding protocol and project-owned errors.
- sentence_transformer: lazy local multilingual embeddings.
"""

from __future__ import annotations

from . import embeddings_chunks as chunks
from . import embeddings_contracts as contracts
from . import embeddings_sentence_transformer as sentence_transformer

__all__ = ["chunks", "contracts", "sentence_transformer"]
