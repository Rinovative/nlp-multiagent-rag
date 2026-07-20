"""
===============================================================================
orchestration_rag.py
===============================================================================
Coordinate the three-agent RAG request graph with LangGraph.

Responsibilities:
  - Load conversation memory and retrieve relevant vector records.
  - Generate one attributed answer through the provider router.
  - Derive ordered source references from the retrieved record metadata.
  - Persist the completed user and assistant exchange.

Design principles:
  - Keep public graph state typed and every session identifier explicit.
  - Propagate domain errors without translating them at orchestration time.

Boundaries:
  - Nodes delegate retrieval, generation, and persistence to injected agents.
  - Does not render source content or construct providers, stores, or sessions.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from src import agents, providers

__all__ = ["RAGChatbot", "RAGResult", "RAGState", "SourceReference"]


@dataclass(frozen=True)
class SourceReference:
    """Identify one retrieved document location without exposing chunk text.

    Parameters
    ----------
    document_name
        Normalized document title or filename safe for plain-text rendering.
    page_number
        Positive one-based page number, or ``None`` when metadata has none.
    """

    document_name: str
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Reject empty names and invalid page numbers."""

        if not self.document_name.strip():
            raise ValueError("document_name must be non-empty")
        if self.page_number is not None and (
            isinstance(self.page_number, bool)
            or not isinstance(self.page_number, int)
            or self.page_number <= 0
        ):
            raise ValueError("page_number must be a positive integer or None")


@dataclass(frozen=True)
class RAGResult:
    """Combine one attributed generation result with ranked source references.

    Parameters
    ----------
    generation
        Provider-neutral answer and actual provider attribution.
    sources
        Deduplicated source references in retrieval-relevance order.
    """

    generation: providers.contracts.GenerationResult
    sources: tuple[SourceReference, ...]

    @property
    def answer(self) -> str:
        """Return the normalized generated answer."""

        return self.generation.answer

    @property
    def provider_id(self) -> str:
        """Return the provider that actually generated the answer."""

        return self.generation.provider_id

    @property
    def model_id(self) -> str:
        """Return the model that actually generated the answer."""

        return self.generation.model_id

    @property
    def fallback_occurred(self) -> bool:
        """Return whether the router used its single controlled fallback."""

        return self.generation.fallback_occurred

    @property
    def fallback_reason(self) -> str | None:
        """Return the machine-readable fallback reason when applicable."""

        return self.generation.fallback_reason


def _source_references(records: list[dict[str, Any]]) -> tuple[SourceReference, ...]:
    """Derive unique safe source references without exposing chunk text."""

    references: list[SourceReference] = []
    seen: set[tuple[str, int | None]] = set()
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            continue
        raw_name = metadata.get("document_title") or metadata.get("file_name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        document_name = " ".join(raw_name.split())[:200]
        raw_page = metadata.get("page_number")
        page_number = (
            raw_page
            if isinstance(raw_page, int)
            and not isinstance(raw_page, bool)
            and raw_page > 0
            else None
        )
        key = (document_name, page_number)
        if key in seen:
            continue
        seen.add(key)
        references.append(SourceReference(document_name, page_number))
    return tuple(references)


class RAGState(TypedDict, total=False):
    """Describe the public typed state exchanged by fixed LangGraph nodes.

    Attributes
    ----------
    chat_id
        Explicit identifier for memory and quota isolation.
    user_input
        Validated current question.
    history
        Existing canonical messages for the chat.
    retrieved_records
        Ranked records returned by the Retriever agent.
    generation_result
        Normalized attributed answer returned by the Generator agent.
    """

    chat_id: str
    user_input: str
    history: list[dict[str, str]]
    retrieved_records: list[dict[str, Any]]
    generation_result: providers.contracts.GenerationResult


class _CompiledRAGGraph(Protocol):
    """Narrow interface used from LangGraph's generic compiled graph."""

    def invoke(self, state: RAGState) -> RAGState:
        """Execute one graph request and return its completed state."""

        ...


class RAGChatbot:
    """Coordinate Memory, Retriever, and Generator agents for one chat ID.

    Parameters
    ----------
    retriever_agent
        Agent that embeds the current question and searches the active store.
    generator_agent
        Agent that builds a bounded request and invokes the provider router.
    memory_agent
        Agent that reads and appends explicitly session-keyed history.

    Notes
    -----
    Construction compiles a fixed graph in the order memory, retrieval,
    generation, and memory storage. Domain errors propagate to the caller.
    """

    def __init__(
        self,
        *,
        retriever_agent: agents.retriever.RetrieverAgent,
        generator_agent: agents.generator.GeneratorAgent,
        memory_agent: agents.memory.MemoryAgent,
    ) -> None:
        """Compile the fixed memory, retrieval, generation, and storage graph."""

        self.retriever_agent = retriever_agent
        self.generator_agent = generator_agent
        self.memory_agent = memory_agent

        graph_builder = StateGraph(RAGState)
        graph_builder.add_node("get_memory", self._get_memory)
        graph_builder.add_node("retrieve", self._retrieve)
        graph_builder.add_node("generate", self._generate)
        graph_builder.add_node("store_memory", self._store_memory)
        graph_builder.add_edge(START, "get_memory")
        graph_builder.add_edge("get_memory", "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", "store_memory")
        graph_builder.add_edge("store_memory", END)
        self.graph = cast(_CompiledRAGGraph, graph_builder.compile())

    def _get_memory(self, state: RAGState) -> RAGState:
        return {"history": self.memory_agent.get_history(state["chat_id"])}

    def _retrieve(self, state: RAGState) -> RAGState:
        return {
            "retrieved_records": self.retriever_agent.retrieve_documents(
                state["user_input"], state.get("history", [])
            )
        }

    def _generate(self, state: RAGState) -> RAGState:
        return {
            "generation_result": self.generator_agent.generate_answer(
                state["user_input"],
                state.get("retrieved_records", []),
                state.get("history", []),
                session_id=state["chat_id"],
            )
        }

    def _store_memory(self, state: RAGState) -> RAGState:
        chat_id = state["chat_id"]
        result = state["generation_result"]
        self.memory_agent.add_message(chat_id, "user", state["user_input"])
        self.memory_agent.add_message(chat_id, "assistant", result.answer)
        return {}

    def process_user_input(self, user_input: str, *, chat_id: str) -> RAGResult:
        """Run the complete graph for one validated question.

        Parameters
        ----------
        user_input
            Non-empty current question.
        chat_id
            Non-empty identifier used for memory and provider quota scope.

        Returns
        -------
        RAGResult
            Attributed answer and ranked, deduplicated source references.

        Raises
        ------
        ValueError
            If the question or chat identifier is empty or invalid.

        Notes
        -----
        The user and assistant messages are appended only after generation
        succeeds; upstream retrieval or provider errors leave history unchanged.
        """

        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")
        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("chat_id must be a non-empty string")
        initial_state: RAGState = {
            "chat_id": chat_id,
            "user_input": user_input.strip(),
        }
        result = self.graph.invoke(initial_state)
        return RAGResult(
            generation=result["generation_result"],
            sources=_source_references(result.get("retrieved_records", [])),
        )
