"""
===============================================================================
agents_memory.py
===============================================================================
Expose conversation memory operations to the RAG graph.

Responsibilities:
  - Read session-keyed histories through the conversation-store contract.
  - Append validated user and assistant messages for graph nodes.

Design principles:
  - Keep graph-facing memory operations explicit and session keyed.

Boundaries:
  - Owns no persistence format or storage connection.
  - Does not choose retention or concurrency policies.
===============================================================================
"""

from __future__ import annotations

from src import memory

__all__ = ["MemoryAgent"]


class MemoryAgent:
    """Adapt one conversation store to graph-facing memory operations.

    Parameters
    ----------
    conversation_store
        Store that owns validation, retention, and concurrency behaviour.
    """

    def __init__(self, conversation_store: memory.contracts.ConversationStore) -> None:
        """Create an agent over one injected conversation store."""

        self.conversation_store = conversation_store

    def get_history(self, chat_id: str) -> list[dict[str, str]]:
        """Return recent history for one explicit chat identifier.

        Parameters
        ----------
        chat_id
            Identifier whose isolated history is requested.

        Returns
        -------
        list of dict
            Ordered canonical messages returned by the injected store.
        """

        return self.conversation_store.get_history(chat_id)

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        """Append one validated conversation message.

        Parameters
        ----------
        chat_id
            Identifier of the history to mutate.
        role
            Canonical ``user`` or ``assistant`` role.
        content
            Non-empty message text.

        Raises
        ------
        TypeError
            If a value has an invalid type.
        ValueError
            If the identifier, role, or content violates the store contract.
        """

        self.conversation_store.append(chat_id, role, content)
