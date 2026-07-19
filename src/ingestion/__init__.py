"""PDF loading, preprocessing, chunking, and ingestion coordination.

Provides:
- chunker: canonical deterministic chunk schema.
- loader: PDF byte extraction.
- preprocessing: structural PDF normalization.
- processor: end-to-end ingestion coordination.
"""

from __future__ import annotations

from . import ingestion_chunker as chunker
from . import ingestion_loader as loader
from . import ingestion_preprocessing as preprocessing
from . import ingestion_processor as processor

__all__ = ["chunker", "loader", "preprocessing", "processor"]
