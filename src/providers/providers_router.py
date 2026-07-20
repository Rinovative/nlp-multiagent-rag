"""
===============================================================================
providers_router.py
===============================================================================
Select generation providers and enforce at most one bounded fallback.

Responsibilities:
  - Implement ``auto``, ``huggingface``, and ``openai`` modes deterministically.
  - Authorize every OpenAI request through a hard-quota backend.
  - Reconcile successful usage and classify permitted fallback conditions.
  - Preserve actual provider and model attribution in every returned result.

Design principles:
  - Fail closed for paid usage and retain ambiguous token reservations.
  - Never retry recursively or invoke more than one fallback provider.

Boundaries:
  - Does not construct provider clients or implement provider SDK calls.
  - Does not release reservations after potentially billable ambiguous failures.
===============================================================================
"""

from __future__ import annotations

import logging
from typing import Literal

from src import quota

from . import providers_contracts as contracts

__all__ = ["GenerationRouter"]

GenerationMode = Literal["auto", "huggingface", "openai"]
_LOGGER = logging.getLogger(__name__)


class GenerationRouter:
    """Route generation deterministically with fail-closed OpenAI authorization.

    Parameters
    ----------
    mode
        ``auto``, ``huggingface``, or ``openai`` routing mode.
    free_provider
        Hugging Face route available without Redis quota authorization.
    openai_provider
        Optional paid provider; absence makes ``auto`` use the free route directly.
    quota_backend
        Backend required before every OpenAI call.
    openai_fallback_enabled
        Whether explicit ``openai`` mode may use the free route for allowed causes.

    Notes
    -----
    ``auto`` permits one free fallback for exhausted or unavailable quota and for
    OpenAI rate-limit or temporary errors. Authentication, request, safety, and
    malformed-response errors do not fall back. Ambiguous reservations are kept.
    """

    def __init__(
        self,
        *,
        mode: GenerationMode,
        free_provider: contracts.GenerationProvider,
        openai_provider: contracts.GenerationProvider | None = None,
        quota_backend: quota.contracts.QuotaBackend | None = None,
        openai_fallback_enabled: bool = False,
    ) -> None:
        """Create a router from explicit providers, mode, and quota policy."""

        if mode not in {"auto", "huggingface", "openai"}:
            raise ValueError("mode must be auto, huggingface, or openai")
        self._mode = mode
        self._free_provider = free_provider
        self._openai_provider = openai_provider
        self._quota_backend = quota_backend
        self._openai_fallback_enabled = openai_fallback_enabled

    @property
    def mode(self) -> GenerationMode:
        """Return the configured routing mode."""

        return self._mode

    def _free(
        self, request: contracts.GenerationRequest, *, reason: str | None = None
    ) -> contracts.GenerationResult:
        _LOGGER.info(
            "generation_route_selected provider=%s model=%s "
            "fallback_reason=%s provider_call_attempted=false",
            self._free_provider.provider_id,
            self._free_provider.model_id,
            reason or "none",
        )
        try:
            result = self._free_provider.generate(request)
        except contracts.GenerationError as exc:
            _LOGGER.warning(
                "generation_route_failed provider=%s model=%s "
                "error_category=%s fallback_reason=%s "
                "provider_call_attempted=true",
                self._free_provider.provider_id,
                self._free_provider.model_id,
                exc.error_category,
                reason or "none",
            )
            if reason is not None:
                raise contracts.GenerationFallbackError(
                    provider_id=self._free_provider.provider_id,
                    model_id=self._free_provider.model_id,
                    fallback_reason=reason,
                    provider_error=exc,
                ) from exc
            raise
        return result if reason is None else result.with_fallback(reason)

    def _log_quota_denial(self, *, reason: str) -> None:
        """Record a secret-safe denial before any OpenAI provider call."""

        model_id = (
            "unconfigured"
            if self._openai_provider is None
            else self._openai_provider.model_id
        )
        _LOGGER.info(
            "openai_quota_authorization_denied provider=openai model=%s "
            "fallback_reason=%s provider_call_attempted=false",
            model_id,
            reason,
        )

    @staticmethod
    def _safe_release(
        backend: quota.contracts.QuotaBackend,
        reservation: quota.contracts.QuotaReservation,
    ) -> None:
        try:
            backend.release(reservation)
        except quota.contracts.QuotaUnavailableError:
            # Leaving the conservative reservation in place is financially safe.
            return

    @staticmethod
    def _safe_reconcile(
        backend: quota.contracts.QuotaBackend,
        reservation: quota.contracts.QuotaReservation,
        *,
        actual_tokens: int,
    ) -> None:
        try:
            backend.reconcile(reservation, actual_tokens=actual_tokens)
        except quota.contracts.QuotaUnavailableError:
            # The original conservative reservation remains the hard upper bound.
            return

    def _quota_fallback_allowed(self) -> bool:
        return self.mode == "auto" or self._openai_fallback_enabled

    def _generate_openai(
        self, request: contracts.GenerationRequest, *, session_id: str
    ) -> contracts.GenerationResult:
        if self._openai_provider is None:
            raise contracts.GenerationConfigurationError(
                "OpenAI generation is selected but OPENAI_API_KEY is not configured."
            )
        if self._quota_backend is None:
            self._log_quota_denial(reason="openai_quota_unavailable")
            if self._quota_fallback_allowed():
                return self._free(request, reason="openai_quota_unavailable")
            raise quota.contracts.QuotaUnavailableError(
                "OpenAI generation requires an available Redis quota backend."
            )

        try:
            reservation = self._quota_backend.reserve(
                session_id=session_id,
                estimated_tokens=request.estimated_total_tokens,
            )
        except quota.contracts.QuotaExhaustedError as exc:
            self._log_quota_denial(reason=exc.reason)
            if self._quota_fallback_allowed():
                return self._free(request, reason=exc.reason)
            raise
        except quota.contracts.QuotaUnavailableError:
            self._log_quota_denial(reason="openai_quota_unavailable")
            if self._quota_fallback_allowed():
                return self._free(request, reason="openai_quota_unavailable")
            raise

        try:
            result = self._openai_provider.generate(request)
        except contracts.GenerationRateLimitError:
            self._safe_release(self._quota_backend, reservation)
            if self._quota_fallback_allowed():
                return self._free(request, reason="openai_rate_limited")
            raise
        except contracts.GenerationTemporaryError:
            # A timeout or broken connection can occur after billable generation.
            # Retaining the conservative reservation preserves the hard cap.
            if self._quota_fallback_allowed():
                return self._free(request, reason="openai_temporarily_unavailable")
            raise
        except contracts.GenerationError:
            # Without provider usage metadata, retaining the reservation is safer
            # than assuming a rejected or malformed response was not billable.
            raise

        actual_tokens = result.usage.total_tokens
        self._safe_reconcile(
            self._quota_backend,
            reservation,
            actual_tokens=(
                request.estimated_total_tokens
                if actual_tokens is None
                else actual_tokens
            ),
        )
        return result

    def generate(
        self,
        request: contracts.GenerationRequest,
        *,
        session_id: str,
    ) -> contracts.GenerationResult:
        """Generate through the configured route with at most one fallback.

        Parameters
        ----------
        request
            Validated bounded generation request.
        session_id
            Non-empty identifier hashed by the Redis quota backend for throttling.

        Returns
        -------
        contracts.GenerationResult
            Actual provider result, including fallback attribution when applicable.

        Raises
        ------
        ValueError
            If ``session_id`` is empty or invalid.
        contracts.GenerationError
            If configuration or provider execution fails without allowed fallback.
        quota.contracts.QuotaError
            If OpenAI authorization fails closed without allowed fallback.

        Notes
        -----
        Successful OpenAI usage is reconciled to provider-reported totals when
        available. Rate-limit failures release tokens; potentially billable
        temporary or malformed responses retain the conservative reservation.
        """

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if self.mode == "huggingface":
            return self._free(request)
        if self.mode == "auto" and self._openai_provider is None:
            return self._free(request)
        return self._generate_openai(request, session_id=session_id)
