"""PDF RAG Assistant application domains.

Provides:
- agents: Retriever, Generator, and Memory roles.
- application: session lifecycle and dependency composition.
- cli: import-free executable administration modules.
- configuration: canonical validated settings.
- embeddings: local document and query vectors.
- ingestion: PDF extraction, preprocessing, and chunking.
- memory: conversation storage.
- orchestration: LangGraph request coordination.
- providers: answer providers and deterministic routing.
- quota: hard OpenAI usage enforcement.
- vectorstore: FAISS retrieval and persistence.
"""

from __future__ import annotations

from . import agents
from . import application
from . import cli
from . import configuration
from . import embeddings
from . import ingestion
from . import memory
from . import orchestration
from . import providers
from . import quota
from . import vectorstore

__all__ = [
    "agents",
    "application",
    "cli",
    "configuration",
    "embeddings",
    "ingestion",
    "memory",
    "orchestration",
    "providers",
    "quota",
    "vectorstore",
]
