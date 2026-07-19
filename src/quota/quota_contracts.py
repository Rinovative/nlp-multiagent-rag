"""
===============================================================================
quota_contracts.py
===============================================================================
Define hard-quota state, periods, failures, and backend operations.

Responsibilities:
  - Model immutable limits, UTC periods, reservations, and usage snapshots.
  - Specify the backend operations required by routing and administration.

Design principles:
  - Make every authorization and reconciliation state explicit.
  - Represent all rolling boundaries with timezone-aware UTC values.

Boundaries:
  - Contains no Redis commands, hidden local-time clocks, or provider calls.
  - Does not grant distributed guarantees to every backend implementation.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

__all__ = [
    "QuotaBackend",
    "QuotaError",
    "QuotaExhaustedError",
    "QuotaLimits",
    "QuotaPeriods",
    "QuotaReservation",
    "QuotaUnavailableError",
    "QuotaUsageSnapshot",
]


class QuotaError(RuntimeError):
    """Represent a project-owned quota failure safe for UI display."""


class QuotaUnavailableError(QuotaError):
    """Indicate that quota state is unavailable and paid usage must fail closed."""


class QuotaExhaustedError(QuotaError):
    """Indicate that a configured hard limit prevents an OpenAI request.

    Parameters
    ----------
    reason
        Machine-readable limit or availability reason used by routing.
    """

    def __init__(self, reason: str) -> None:
        """Create a UI-safe error retaining one machine-readable reason."""

        super().__init__("The configured OpenAI application allowance is unavailable.")
        self.reason = reason


@dataclass(frozen=True)
class QuotaLimits:
    """Represent immutable owner-controlled hard limits.

    Parameters
    ----------
    daily_requests
        Positive application-wide request limit per UTC day.
    monthly_requests
        Positive application-wide request limit per UTC month.
    daily_tokens
        Positive reserved-token limit per UTC day.
    monthly_tokens
        Positive reserved-token limit per UTC month.
    session_requests
        Positive per-session request limit in each fixed window.
    session_window_seconds
        Positive duration of the fixed per-session window.

    Raises
    ------
    ValueError
        If values are non-positive or monthly limits are below daily limits.
    """

    daily_requests: int
    monthly_requests: int
    daily_tokens: int
    monthly_tokens: int
    session_requests: int
    session_window_seconds: int

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.monthly_requests < self.daily_requests:
            raise ValueError("monthly_requests must be at least daily_requests")
        if self.monthly_tokens < self.daily_tokens:
            raise ValueError("monthly_tokens must be at least daily_tokens")


@dataclass(frozen=True)
class QuotaPeriods:
    """Represent immutable UTC period identifiers used in quota keys.

    Parameters
    ----------
    day
        Calendar-day identifier in ``YYYY-MM-DD`` form.
    month
        Calendar-month identifier in ``YYYY-MM`` form.
    session_window
        Integer fixed-window identifier represented as text.
    """

    day: str
    month: str
    session_window: str

    @classmethod
    def at(cls, now: datetime, *, session_window_seconds: int) -> "QuotaPeriods":
        """Build UTC day, month, and fixed session-window identifiers.

        Parameters
        ----------
        now
            Timezone-aware instant converted to UTC.
        session_window_seconds
            Positive fixed-window duration used for session throttling.

        Returns
        -------
        QuotaPeriods
            Immutable identifiers for the instant's active periods.

        Raises
        ------
        ValueError
            If ``now`` is naive.
        """

        if now.tzinfo is None:
            raise ValueError("quota timestamps must be timezone-aware")
        current = now.astimezone(UTC)
        window_start = int(current.timestamp()) // session_window_seconds
        return cls(
            day=current.strftime("%Y-%m-%d"),
            month=current.strftime("%Y-%m"),
            session_window=str(window_start),
        )


@dataclass(frozen=True)
class QuotaReservation:
    """Represent an immutable conservative reservation for one OpenAI call.

    Parameters
    ----------
    reservation_id
        Opaque identifier used for idempotent reconciliation or release.
    reserved_tokens
        Conservative token count charged before provider invocation.
    periods
        UTC periods whose aggregate counters contain the reservation.
    """

    reservation_id: str
    reserved_tokens: int
    periods: QuotaPeriods


@dataclass(frozen=True)
class QuotaUsageSnapshot:
    """Represent an immutable owner-facing aggregate usage snapshot.

    Parameters
    ----------
    enabled
        Current authorization switch, or ``None`` when limits are unconfigured.
    limits
        Active numeric limits, or ``None`` when unconfigured.
    periods
        UTC periods represented by the counters.
    daily_requests
        Application request count in the active UTC day.
    monthly_requests
        Application request count in the active UTC month.
    daily_tokens
        Reserved or reconciled token count in the active UTC day.
    monthly_tokens
        Reserved or reconciled token count in the active UTC month.
    """

    enabled: bool | None
    limits: QuotaLimits | None
    periods: QuotaPeriods
    daily_requests: int = 0
    monthly_requests: int = 0
    daily_tokens: int = 0
    monthly_tokens: int = 0


class QuotaBackend(Protocol):
    """Abstract hard-quota operations used by routing and administration.

    Implementations must authorize and reserve a request as one atomic state
    transition within their documented scope, fail closed on uncertain state,
    and make reconciliation and release idempotent for each reservation.
    """

    def inspect(self, *, now: datetime | None = None) -> QuotaUsageSnapshot:
        """Return limits and aggregate usage for the active periods.

        Parameters
        ----------
        now
            Optional timezone-aware instant; defaults to the current UTC time.

        Returns
        -------
        QuotaUsageSnapshot
            Owner-visible settings and aggregate application counters.
        """

        ...

    def set_limits(self, limits: QuotaLimits) -> None:
        """Replace all owner-controlled limits without implicitly enabling usage.

        Parameters
        ----------
        limits
            Complete validated limit set to persist.
        """

        ...

    def set_enabled(self, enabled: bool) -> None:
        """Change the immediate authorization switch.

        Parameters
        ----------
        enabled
            Whether subsequent paid-provider reservations may be authorized.

        Raises
        ------
        QuotaUnavailableError
            If numeric limits have not been configured or storage is unavailable.
        """

        ...

    def reserve(
        self,
        *,
        session_id: str,
        estimated_tokens: int,
        now: datetime | None = None,
    ) -> QuotaReservation:
        """Atomically authorize counters and reserve one request.

        Parameters
        ----------
        session_id
            Non-empty identifier used for per-session throttling.
        estimated_tokens
            Positive conservative input-plus-maximum-output token count.
        now
            Optional timezone-aware instant; defaults to the current UTC time.

        Returns
        -------
        QuotaReservation
            Reservation to reconcile or release after provider execution.

        Raises
        ------
        QuotaExhaustedError
            If authorization is disabled or a projected hard limit is exceeded.
        QuotaUnavailableError
            If authorization cannot be decided reliably.
        """

        ...

    def reconcile(self, reservation: QuotaReservation, *, actual_tokens: int) -> None:
        """Replace reserved token usage with reported actual usage idempotently.

        Parameters
        ----------
        reservation
            Previously authorized reservation.
        actual_tokens
            Non-negative provider-reported total token count.

        Notes
        -----
        Request counters remain consumed.
        """

        ...

    def release(self, reservation: QuotaReservation) -> None:
        """Release tokens for a definitely unbilled reservation idempotently.

        Parameters
        ----------
        reservation
            Previously authorized reservation whose tokens can be released safely.

        Notes
        -----
        Request counters remain consumed.
        """

        ...
