"""
===============================================================================
providers_contracts.py
===============================================================================
Define provider-neutral generation values, interfaces, and failures.

Responsibilities:
  - Define immutable messages, requests, usage, and attributed results.
  - Specify the generation-provider capability required by routing.
  - Provide precise project-owned generation exception categories.

Design principles:
  - Keep routing and orchestration independent of SDK response types.
  - Validate request and attribution values at construction time.

Boundaries:
  - Contains no client construction, provider calls, or fallback policy.
  - Does not expose third-party exception details at the UI boundary.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

__all__ = [
    "GenerationAuthenticationError",
    "GenerationConfigurationError",
    "GenerationError",
    "GenerationInvalidRequestError",
    "GenerationMessage",
    "GenerationProvider",
    "GenerationRateLimitError",
    "GenerationRequest",
    "GenerationResponseError",
    "GenerationResult",
    "GenerationSafetyError",
    "GenerationTemporaryError",
    "GenerationUsage",
]


class GenerationError(RuntimeError):
    """Represent a project-owned generation failure safe for UI display."""


class GenerationConfigurationError(GenerationError):
    """Indicate that a requested generation route lacks required configuration."""


class GenerationAuthenticationError(GenerationError):
    """Indicate rejected provider credentials or permissions without leaking them."""


class GenerationRateLimitError(GenerationError):
    """Indicate a provider-owned rate, credit, or capacity rejection."""


class GenerationTemporaryError(GenerationError):
    """Indicate a transient timeout, connection, overload, or server failure."""


class GenerationInvalidRequestError(GenerationError):
    """Indicate invalid provider configuration or request shape."""


class GenerationSafetyError(GenerationError):
    """Indicate that a provider rejected generation for safety reasons."""


class GenerationResponseError(GenerationError):
    """Indicate an unusable malformed or empty provider response."""


@dataclass(frozen=True)
class GenerationMessage:
    """Represent one immutable provider-neutral chat message.

    Parameters
    ----------
    role
        Canonical ``system``, ``user``, or ``assistant`` role.
    content
        Non-empty message text.

    Raises
    ------
    ValueError
        If the message content is empty.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Generation message content must not be empty.")


@dataclass(frozen=True)
class GenerationRequest:
    """Represent an immutable bounded generation-provider request.

    Parameters
    ----------
    messages
        Non-empty ordered provider-neutral chat messages.
    max_output_tokens
        Positive upper bound requested for completion tokens.
    estimated_input_tokens
        Positive conservative input-token estimate used for quota reservation.

    Raises
    ------
    ValueError
        If messages are empty or either token bound is not positive.
    """

    messages: tuple[GenerationMessage, ...]
    max_output_tokens: int
    estimated_input_tokens: int

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("Generation requests need at least one message.")
        if self.max_output_tokens <= 0 or self.estimated_input_tokens <= 0:
            raise ValueError("Generation token bounds must be positive.")

    @property
    def estimated_total_tokens(self) -> int:
        """Return the conservative amount reserved before an OpenAI call."""

        return self.estimated_input_tokens + self.max_output_tokens


@dataclass(frozen=True)
class GenerationUsage:
    """Represent immutable normalized usage reported by a provider.

    Parameters
    ----------
    input_tokens
        Reported prompt-token count, or ``None`` when unavailable.
    output_tokens
        Reported completion-token count, or ``None`` when unavailable.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        """Return total reported tokens only when both components are known."""

        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class GenerationResult:
    """Represent an immutable answer with actual-provider attribution.

    Parameters
    ----------
    answer
        Non-empty normalized answer text.
    provider_id
        Stable identifier of the provider that produced the answer.
    model_id
        Configured identifier of the model that produced the answer.
    usage
        Normalized provider-reported usage, which may be partially unknown.
    fallback_occurred
        Whether the router used its single permitted fallback.
    fallback_reason
        Machine-readable reason paired with a true fallback flag.

    Raises
    ------
    ValueError
        If attribution is empty or fallback fields are inconsistent.
    """

    answer: str
    provider_id: str
    model_id: str
    usage: GenerationUsage
    fallback_occurred: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("Generation results need non-empty answer text.")
        if not self.provider_id.strip() or not self.model_id.strip():
            raise ValueError("Generation results need provider and model identifiers.")
        if self.fallback_occurred != (self.fallback_reason is not None):
            raise ValueError("Fallback status and reason must be recorded together.")

    def with_fallback(self, reason: str) -> "GenerationResult":
        """Return a copy marked as the router's single fallback result.

        Parameters
        ----------
        reason
            Non-empty machine-readable fallback reason.

        Returns
        -------
        GenerationResult
            Immutable copy with consistent fallback fields.

        Raises
        ------
        ValueError
            If ``reason`` is empty.
        """

        if not reason.strip():
            raise ValueError("fallback reason must be non-empty")
        return replace(self, fallback_occurred=True, fallback_reason=reason)


class GenerationProvider(Protocol):
    """Abstract the normalized answer-generation capability.

    Implementations must make at most the calls they document, return actual
    provider and model attribution, and translate SDK failures into the precise
    project-owned :class:`GenerationError` hierarchy without leaking secrets.
    """

    @property
    def provider_id(self) -> str:
        """Return the stable provider identifier."""

        ...

    @property
    def model_id(self) -> str:
        """Return the configured generation model identifier."""

        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one normalized provider-attributed answer.

        Parameters
        ----------
        request
            Validated bounded provider-neutral request.

        Returns
        -------
        GenerationResult
            Normalized answer and reported usage from this provider.

        Raises
        ------
        GenerationError
            If configuration, provider execution, or response validation fails.
        """

        ...
