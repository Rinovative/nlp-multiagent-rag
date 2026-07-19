"""
===============================================================================
quota_memory.py
===============================================================================
Provide deterministic process-local quota behavior for tests and verification.

Responsibilities:
  - Mirror limit, reservation, reconciliation, and rollover policy in memory.
  - Apply request and token mutations under one process-local lock.

Design principles:
  - Serialize state transitions under one re-entrant lock.
  - Keep reconciliation and release idempotent per reservation.

Boundaries:
  - Is not distributed public-demo financial protection.
  - Does not persist configuration or counters across process restarts.
===============================================================================
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from threading import RLock

from . import quota_contracts as contracts

__all__ = ["InMemoryQuotaBackend"]


class InMemoryQuotaBackend:
    """Provide a thread-safe process-local reference quota backend.

    Notes
    -----
    All state transitions are serialized by one re-entrant lock. Guarantees apply
    only within the current process; state is neither distributed nor persistent.
    The backend starts disabled and without configured limits.
    """

    def __init__(self) -> None:
        """Create an initially disabled and unconfigured reference backend."""

        self._limits: contracts.QuotaLimits | None = None
        self._enabled = False
        self._requests: dict[tuple[str, str], int] = {}
        self._tokens: dict[tuple[str, str], int] = {}
        self._session_requests: dict[tuple[str, str], int] = {}
        self._reservations: dict[str, tuple[str, int, contracts.QuotaPeriods]] = {}
        self._lock = RLock()

    @staticmethod
    def _now(now: datetime | None) -> datetime:
        return datetime.now(UTC) if now is None else now

    @staticmethod
    def _session_hash(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    def set_limits(self, limits: contracts.QuotaLimits) -> None:
        """Replace active numeric limits without implicitly enabling OpenAI.

        Parameters
        ----------
        limits
            Complete immutable limit set retained in process memory.
        """

        with self._lock:
            self._limits = limits

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable authorization immediately within this process.

        Parameters
        ----------
        enabled
            Whether subsequent reservations may be authorized.

        Raises
        ------
        contracts.QuotaUnavailableError
            If limits have not been configured.
        """

        with self._lock:
            if self._limits is None:
                raise contracts.QuotaUnavailableError(
                    "OpenAI quota limits must be configured before enabling usage."
                )
            self._enabled = bool(enabled)

    def _periods(self, now: datetime | None) -> contracts.QuotaPeriods:
        if self._limits is None:
            raise contracts.QuotaUnavailableError(
                "OpenAI quota limits have not been configured."
            )
        return contracts.QuotaPeriods.at(
            self._now(now), session_window_seconds=self._limits.session_window_seconds
        )

    def inspect(self, *, now: datetime | None = None) -> contracts.QuotaUsageSnapshot:
        """Return current aggregate counters for the active UTC periods.

        Parameters
        ----------
        now
            Optional timezone-aware instant; defaults to current UTC time.

        Returns
        -------
        contracts.QuotaUsageSnapshot
            Current process-local limits, switch, periods, and counters.
        """

        with self._lock:
            if self._limits is None:
                current = self._now(now)
                periods = contracts.QuotaPeriods.at(
                    current, session_window_seconds=3600
                )
                return contracts.QuotaUsageSnapshot(
                    enabled=None, limits=None, periods=periods
                )
            periods = self._periods(now)
            return contracts.QuotaUsageSnapshot(
                enabled=self._enabled,
                limits=self._limits,
                periods=periods,
                daily_requests=self._requests.get(("day", periods.day), 0),
                monthly_requests=self._requests.get(("month", periods.month), 0),
                daily_tokens=self._tokens.get(("day", periods.day), 0),
                monthly_tokens=self._tokens.get(("month", periods.month), 0),
            )

    def reserve(
        self,
        *,
        session_id: str,
        estimated_tokens: int,
        now: datetime | None = None,
    ) -> contracts.QuotaReservation:
        """Atomically authorize counters and reserve conservative tokens.

        Parameters
        ----------
        session_id
            Non-empty identifier hashed for the fixed-window request counter.
        estimated_tokens
            Positive conservative token count charged to day and month counters.
        now
            Optional timezone-aware instant; defaults to current UTC time.

        Returns
        -------
        contracts.QuotaReservation
            Opaque reservation tied to the charged UTC periods.

        Raises
        ------
        ValueError
            If the session identifier or token estimate is invalid.
        contracts.QuotaExhaustedError
            If disabled or a projected hard limit would be exceeded.
        contracts.QuotaUnavailableError
            If limits have not been configured.
        """

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")
        with self._lock:
            periods = self._periods(now)
            limits = self._limits
            assert limits is not None
            if not self._enabled:
                raise contracts.QuotaExhaustedError("openai_disabled")

            day_request_key = ("day", periods.day)
            month_request_key = ("month", periods.month)
            day_token_key = ("day", periods.day)
            month_token_key = ("month", periods.month)
            session_key = (self._session_hash(session_id), periods.session_window)
            checks = (
                (
                    self._requests.get(day_request_key, 0) + 1,
                    limits.daily_requests,
                    "daily_requests_exhausted",
                ),
                (
                    self._requests.get(month_request_key, 0) + 1,
                    limits.monthly_requests,
                    "monthly_requests_exhausted",
                ),
                (
                    self._tokens.get(day_token_key, 0) + estimated_tokens,
                    limits.daily_tokens,
                    "daily_tokens_exhausted",
                ),
                (
                    self._tokens.get(month_token_key, 0) + estimated_tokens,
                    limits.monthly_tokens,
                    "monthly_tokens_exhausted",
                ),
                (
                    self._session_requests.get(session_key, 0) + 1,
                    limits.session_requests,
                    "session_requests_exhausted",
                ),
            )
            for projected, limit, reason in checks:
                if projected > limit:
                    raise contracts.QuotaExhaustedError(reason)

            self._requests[day_request_key] = self._requests.get(day_request_key, 0) + 1
            self._requests[month_request_key] = (
                self._requests.get(month_request_key, 0) + 1
            )
            self._tokens[day_token_key] = (
                self._tokens.get(day_token_key, 0) + estimated_tokens
            )
            self._tokens[month_token_key] = (
                self._tokens.get(month_token_key, 0) + estimated_tokens
            )
            self._session_requests[session_key] = (
                self._session_requests.get(session_key, 0) + 1
            )
            reservation = contracts.QuotaReservation(
                reservation_id=uuid.uuid4().hex,
                reserved_tokens=estimated_tokens,
                periods=periods,
            )
            self._reservations[reservation.reservation_id] = (
                "reserved",
                estimated_tokens,
                periods,
            )
            return reservation

    def reconcile(
        self, reservation: contracts.QuotaReservation, *, actual_tokens: int
    ) -> None:
        """Atomically replace reserved tokens with reported actual usage.

        Parameters
        ----------
        reservation
            Previously authorized reservation.
        actual_tokens
            Non-negative provider-reported token total.

        Raises
        ------
        ValueError
            If ``actual_tokens`` is negative.

        Notes
        -----
        Reconciliation is idempotent and leaves request counters consumed.
        """

        if actual_tokens < 0:
            raise ValueError("actual_tokens must not be negative")
        with self._lock:
            stored = self._reservations.get(reservation.reservation_id)
            if stored is None or stored[0] != "reserved":
                return
            _, reserved_tokens, periods = stored
            delta = actual_tokens - reserved_tokens
            for key in (("day", periods.day), ("month", periods.month)):
                self._tokens[key] = max(0, self._tokens.get(key, 0) + delta)
            self._reservations[reservation.reservation_id] = (
                "reconciled",
                actual_tokens,
                periods,
            )

    def release(self, reservation: contracts.QuotaReservation) -> None:
        """Release only token usage for a definitely unbilled reservation.

        Parameters
        ----------
        reservation
            Previously authorized reservation.

        Notes
        -----
        Release is idempotent and leaves request counters consumed.
        """

        with self._lock:
            stored = self._reservations.get(reservation.reservation_id)
            if stored is None or stored[0] != "reserved":
                return
            _, reserved_tokens, periods = stored
            for key in (("day", periods.day), ("month", periods.month)):
                self._tokens[key] = max(0, self._tokens.get(key, 0) - reserved_tokens)
            self._reservations[reservation.reservation_id] = (
                "released",
                0,
                periods,
            )
