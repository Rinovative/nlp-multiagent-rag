"""
===============================================================================
providers_generation_huggingface.py
===============================================================================
Generate answers through Hugging Face Inference Providers.

Responsibilities:
  - Use the supported chat-completion request shape.
  - Normalize answer text, model attribution, and token usage.
  - Translate SDK failures into project-owned exceptions.

Design principles:
  - Make one bounded call and normalize only supported response fields.
  - Create the injected client lazily on first generation.

Boundaries:
  - The injected client factory is invoked only during generation.
  - Does not select routes, retry, or authorize paid-provider usage.
===============================================================================
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from huggingface_hub.errors import (
    HfHubHTTPError,
    InferenceEndpointTimeoutError,
    InferenceTimeoutError,
    OverloadedError,
    ValidationError,
)

from . import providers_contracts as contracts

__all__ = ["HuggingFaceGenerationProvider"]
_LOGGER = logging.getLogger(__name__)


class HuggingFaceGenerationProvider:
    """Run hosted open-model chat completion through an injected client factory.

    Parameters
    ----------
    client_factory
        Callable invoked only when generation runs; it may cache its client.
    model_id
        Non-empty Hugging Face model identifier used for every request.
    temperature
        Sampling temperature forwarded to the chat-completion call.

    Notes
    -----
    The provider does not retry or switch routes and never invokes its client
    factory at construction or import time.
    """

    provider_id = "huggingface"

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        model_id: str,
        temperature: float = 0.2,
    ) -> None:
        """Configure one lazy client factory and hosted model identifier."""

        if not callable(client_factory):
            raise ValueError("client_factory must be callable")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self._client_factory = client_factory
        self._model_id = model_id.strip()
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model identifier."""

        return self._model_id

    @staticmethod
    def _usage(response: Any) -> contracts.GenerationUsage:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return contracts.GenerationUsage(
            input_tokens=(
                input_tokens
                if isinstance(input_tokens, int) and input_tokens >= 0
                else None
            ),
            output_tokens=(
                output_tokens
                if isinstance(output_tokens, int) and output_tokens >= 0
                else None
            ),
        )

    @staticmethod
    def _status_category(status_code: object) -> str:
        if isinstance(status_code, int):
            return f"{status_code // 100}xx"
        return "unknown"

    def _log_failure(
        self,
        error: contracts.GenerationError,
        *,
        status_category: str,
    ) -> None:
        """Record only project-owned, non-sensitive provider diagnostics."""

        _LOGGER.warning(
            "generation_provider_failed provider=%s model=%s "
            "error_category=%s safe_http_status_category=%s "
            "provider_call_attempted=true",
            self.provider_id,
            self.model_id,
            error.error_category,
            status_category,
        )

    @staticmethod
    def _translate_http_error(exc: HfHubHTTPError) -> contracts.GenerationError:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {401, 403}:
            return contracts.GenerationAuthenticationError(
                "Hugging Face rejected the configured inference token or permissions."
            )
        if status_code == 402:
            return contracts.GenerationCreditsError(
                "Hugging Face inference credits are unavailable."
            )
        if status_code == 404:
            return contracts.GenerationModelUnavailableError(
                "The configured Hugging Face model has no available hosted route."
            )
        if status_code == 429:
            return contracts.GenerationRateLimitError(
                "Hugging Face inference is currently rate limited."
            )
        if isinstance(status_code, int) and status_code >= 500:
            return contracts.GenerationTemporaryError(
                "Hugging Face inference is temporarily unavailable."
            )
        return contracts.GenerationInvalidRequestError(
            "Hugging Face rejected the configured model or request."
        )

    def generate(
        self, request: contracts.GenerationRequest
    ) -> contracts.GenerationResult:
        """Generate one answer without retrying or switching providers.

        Parameters
        ----------
        request
            Validated bounded provider-neutral request.

        Returns
        -------
        contracts.GenerationResult
            Normalized answer attributed to Hugging Face and the configured model.

        Raises
        ------
        contracts.GenerationError
            If inference fails or the provider response is empty or malformed.
        """

        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]
        client = self._client_factory()
        _LOGGER.info(
            "generation_client_constructed provider=%s model=%s",
            self.provider_id,
            self.model_id,
        )
        _LOGGER.info(
            "generation_request_attempted provider=%s model=%s "
            "provider_call_attempted=true",
            self.provider_id,
            self.model_id,
        )
        try:
            response = client.chat_completion(
                messages,
                model=self.model_id,
                max_tokens=request.max_output_tokens,
                temperature=self._temperature,
            )
        except (
            InferenceTimeoutError,
            InferenceEndpointTimeoutError,
            OverloadedError,
        ) as exc:
            temporary_error = contracts.GenerationTemporaryError(
                "Hugging Face inference is temporarily unavailable."
            )
            self._log_failure(temporary_error, status_category="unknown")
            raise temporary_error from exc
        except ValidationError as exc:
            invalid_request_error = contracts.GenerationInvalidRequestError(
                "Hugging Face rejected the configured model or request."
            )
            self._log_failure(invalid_request_error, status_category="unknown")
            raise invalid_request_error from exc
        except HfHubHTTPError as exc:
            http_error = self._translate_http_error(exc)
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            self._log_failure(
                http_error,
                status_category=self._status_category(status_code),
            )
            raise http_error from exc
        except (StopIteration, ValueError) as exc:
            model_error = contracts.GenerationModelUnavailableError(
                "The configured Hugging Face model has no available hosted route."
            )
            self._log_failure(model_error, status_category="unknown")
            raise model_error from exc

        try:
            answer = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            response_error = contracts.GenerationResponseError(
                "Hugging Face returned an invalid generation response."
            )
            self._log_failure(response_error, status_category="none")
            raise response_error from exc
        if not isinstance(answer, str) or not answer.strip():
            empty_response_error = contracts.GenerationResponseError(
                "Hugging Face returned an empty generation response."
            )
            self._log_failure(empty_response_error, status_category="none")
            raise empty_response_error
        return contracts.GenerationResult(
            answer=answer.strip(),
            provider_id=self.provider_id,
            model_id=self.model_id,
            usage=self._usage(response),
        )
