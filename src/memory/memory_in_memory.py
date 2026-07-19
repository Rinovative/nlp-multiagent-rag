"""
===============================================================================
memory_in_memory.py
===============================================================================
Store isolated conversation histories in the current process.

Responsibilities:
  - Retain histories keyed by explicit chat identifiers.
  - Trim each history to its configured message limit.

Design principles:
  - Return defensive copies and serialize mutations with a re-entrant lock.

Boundaries:
  - Provides ephemeral application memory, not distributed persistence.
  - Does not coordinate histories across processes or application restarts.
===============================================================================
"""

from __future__ import annotations

import copy
from threading import RLock

from . import memory_contracts as contracts

__all__ = ["InMemoryConversationStore"]


class InMemoryConversationStore:
    """Store thread-safe process-local histories for isolated sessions.

    Parameters
    ----------
    max_history
        Positive maximum number of retained messages per chat identifier.

    Notes
    -----
    Mutations are serialized within one process, returned histories are defensive
    copies, and no state survives a process restart.
    """

    def __init__(self, max_history: int = 10) -> None:
        """Create a store retaining at most the configured messages per chat."""

        if not isinstance(max_history, int) or max_history <= 0:
            raise ValueError("max_history must be a positive integer")
        self.max_history = max_history
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._lock = RLock()

    def get_history(self, chat_id: str) -> list[dict[str, str]]:
        """Return a defensive copy of one session's recent history.

        Parameters
        ----------
        chat_id
            Non-empty identifier whose isolated history is requested.

        Returns
        -------
        list of dict
            Ordered retained messages, or an empty list for an unknown identifier.
        """

        if not isinstance(chat_id, str) or not chat_id:
            raise ValueError("chat_id must be a non-empty string")
        with self._lock:
            return copy.deepcopy(self._histories.get(chat_id, []))

    def append(self, chat_id: str, role: str, content: str) -> None:
        """Append and trim one session's history atomically within this process.

        Parameters
        ----------
        chat_id
            Identifier of the history to mutate.
        role
            Canonical ``user`` or ``assistant`` role.
        content
            Non-empty message content.

        Raises
        ------
        ValueError
            If the identifier, role, or content violates the message contract.
        """

        contracts.validate_message(chat_id, role, content)
        with self._lock:
            history = self._histories.setdefault(chat_id, [])
            history.append({"role": role, "content": content})
            del history[: -self.max_history]

    def clear(self, chat_id: str) -> None:
        """Discard one session's history if it exists.

        Parameters
        ----------
        chat_id
            Identifier whose process-local history is removed.
        """

        with self._lock:
            self._histories.pop(chat_id, None)
