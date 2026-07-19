"""
===============================================================================
providers_generation_openai.py
===============================================================================
Generate answers through the optional OpenAI chat provider.

Responsibilities:
  - Send bounded chat-completion requests through an injected client.
  - Normalize reported usage and answer attribution.
  - Classify authentication, rate, temporary, invalid, and safety failures.

Design principles:
  - Make one bounded call and preserve SDK exceptions only as causes.
  - Create the injected client lazily on first generation.

Boundaries:
  - Quota authorization belongs to the provider router.
  - Does not select routes, retry, or expose raw SDK error details.
===============================================================================
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ContentFilterFinishReasonError,
    InternalServerError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from . import providers_contracts as contracts

__all__ = ["OpenAIGenerationProvider"]


class OpenAIGenerationProvider:
    """Run optional OpenAI chat completion through an injected client factory.

    Parameters
    ----------
    client_factory
        Callable invoked only when generation runs; it may cache its client.
    model_id
        Non-empty OpenAI model identifier used for every request.

    Notes
    -----
    This adapter does not authorize quota usage. The provider router must reserve
    quota before invoking it and reconcile the result afterward.
    """

    provider_id = "openai"

    def __init__(self, client_factory: Callable[[], Any], *, model_id: str) -> None:
        """Configure one lazy client factory and OpenAI model identifier."""

        if not callable(client_factory):
            raise ValueError("client_factory must be callable")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self._client_factory = client_factory
        self._model_id = model_id.strip()

    @property
    def model_id(self) -> str:
        """Return the configured OpenAI model identifier."""

        return self._model_id

    @staticmethod
    def _translate_error(exc: OpenAIError) -> contracts.GenerationError:
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return contracts.GenerationAuthenticationError(
                "OpenAI rejected the configured API key or model permissions."
            )
        if isinstance(exc, RateLimitError):
            return contracts.GenerationRateLimitError(
                "OpenAI is currently rate limited."
            )
        if isinstance(
            exc,
            (APIConnectionError, APITimeoutError, InternalServerError),
        ):
            return contracts.GenerationTemporaryError(
                "OpenAI is temporarily unavailable."
            )
        if isinstance(exc, ContentFilterFinishReasonError):
            return contracts.GenerationSafetyError(
                "OpenAI could not answer this request because of a safety restriction."
            )
        if isinstance(exc, BadRequestError):
            body = getattr(exc, "body", None)
            code = body.get("code") if isinstance(body, dict) else None
            if code in {"content_filter", "safety_violation"}:
                return contracts.GenerationSafetyError(
                    "OpenAI could not answer this request because of a safety restriction."
                )
        if isinstance(exc, (BadRequestError, UnprocessableEntityError)):
            return contracts.GenerationInvalidRequestError(
                "OpenAI rejected the configured model or request."
            )
        return contracts.GenerationResponseError(
            "OpenAI could not complete the generation request."
        )

    @staticmethod
    def _usage(response: Any) -> contracts.GenerationUsage:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return contracts.GenerationUsage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        )

    def generate(
        self, request: contracts.GenerationRequest
    ) -> contracts.GenerationResult:
        """Generate one OpenAI response through the injected client factory.

        Parameters
        ----------
        request
            Validated bounded request already authorized by the caller when needed.

        Returns
        -------
        contracts.GenerationResult
            Normalized answer attributed to OpenAI and the configured model.

        Raises
        ------
        contracts.GenerationError
            If the SDK call fails or the response is empty or malformed.

        Notes
        -----
        The method performs no quota operation and does not retry or fall back.
        """

        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        try:
            response = self._client_factory().chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=request.max_output_tokens,
                temperature=0.2,
            )
        except OpenAIError as exc:
            raise self._translate_error(exc) from exc

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise contracts.GenerationResponseError(
                "OpenAI returned an invalid generation response."
            ) from exc
        if not isinstance(answer, str) or not answer.strip():
            raise contracts.GenerationResponseError(
                "OpenAI returned an empty generation response."
            )
        return contracts.GenerationResult(
            answer=answer.strip(),
            provider_id=self.provider_id,
            model_id=self.model_id,
            usage=self._usage(response),
        )
