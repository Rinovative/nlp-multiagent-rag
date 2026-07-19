"""
===============================================================================
memory_contracts.py
===============================================================================
Define conversation storage behavior and canonical message validation.

Responsibilities:
  - Specify history access, append, and clear operations by chat identifier.
  - Validate canonical user and assistant messages.

Design principles:
  - Require explicit chat identifiers at every boundary.
  - Return plain message mappings independent of a storage implementation.

Boundaries:
  - Contains no storage implementation or graph behavior.
  - Does not prescribe persistence, retention, or locking mechanisms.
===============================================================================
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["ConversationStore", "validate_message"]


class ConversationStore(Protocol):
    """Abstract storage operations required by the Memory agent.

    Implementations must isolate histories by explicit chat identifier, return
    caller-safe message mappings, validate appended messages, and document their
    own retention, persistence, and concurrency guarantees.
    """

    def get_history(self, chat_id: str) -> list[dict[str, str]]:
        """Return the implementation-bounded history for one chat identifier.

        Parameters
        ----------
        chat_id
            Non-empty identifier whose isolated history is requested.

        Returns
        -------
        list of dict
            Ordered canonical message mappings safe for caller mutation.
        """

        ...

    def append(self, chat_id: str, role: str, content: str) -> None:
        """Append one validated message to an isolated history.

        Parameters
        ----------
        chat_id
            Non-empty identifier of the history to mutate.
        role
            Canonical ``user`` or ``assistant`` role.
        content
            Non-empty message text.
        """

        ...

    def clear(self, chat_id: str) -> None:
        """Delete the history for one chat identifier.

        Parameters
        ----------
        chat_id
            Non-empty identifier whose history is removed.
        """

        ...


def validate_message(chat_id: str, role: str, content: str) -> None:
    """Validate one canonical conversation message without storing it.

    Parameters
    ----------
    chat_id
        Non-empty history identifier.
    role
        Canonical ``user`` or ``assistant`` role.
    content
        Non-empty message text.

    Raises
    ------
    ValueError
        If the identifier, role, or content violates the message contract.
    """

    if not isinstance(chat_id, str) or not chat_id:
        raise ValueError("chat_id must be a non-empty string")
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    if not isinstance(content, str) or not content:
        raise ValueError("content must be a non-empty string")
