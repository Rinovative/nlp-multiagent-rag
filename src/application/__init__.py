"""Application composition and session lifecycle.

Provides:
- factory: lazy construction of application components.
- session: isolated browser-session state and upload activation.
"""

from __future__ import annotations

from . import application_factory as factory
from . import application_session as session

__all__ = ["factory", "session"]
