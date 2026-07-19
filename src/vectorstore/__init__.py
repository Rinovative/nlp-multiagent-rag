"""FAISS vector storage and atomic persistence.

Provides:
- faiss: validated in-memory indexes and complete snapshots.
"""

from __future__ import annotations

from . import vectorstore_faiss as faiss

__all__ = ["faiss"]
