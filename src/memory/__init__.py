"""Session-local conversation-memory contracts and storage.

Provides:
- contracts: conversation storage protocol and validation.
- in_memory: isolated process-local histories.
"""

from __future__ import annotations

from . import memory_contracts as contracts
from . import memory_in_memory as in_memory

__all__ = ["contracts", "in_memory"]
