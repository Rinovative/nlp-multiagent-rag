"""
===============================================================================
quota_redis.py
===============================================================================
Enforce distributed OpenAI limits with atomic Redis Lua operations.

Responsibilities:
  - Keep active limits and UTC-window counters in Redis.
  - Atomically authorize requests and reserve conservative token usage.
  - Reconcile or release token reservations after provider completion.
  - Expose owner inspection and transactional settings updates.

Design principles:
  - Authorize multi-key usage atomically and fail closed on uncertain state.
  - Create the Redis client lazily and translate Redis failures.

Boundaries:
  - Stores hashes, counters, period identifiers, and opaque reservation IDs only.
  - Never stores prompts, documents, credentials, or raw session identifiers.
===============================================================================
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, time, timedelta
from typing import Any

import redis
from redis.exceptions import RedisError

from . import quota_contracts as contracts

__all__ = ["RedisQuotaBackend"]

_LIMIT_FIELDS = (
    "daily_requests",
    "monthly_requests",
    "daily_tokens",
    "monthly_tokens",
    "session_requests",
    "session_window_seconds",
)

_SET_ENABLED_SCRIPT = """
for _, field in ipairs({
  'daily_requests', 'monthly_requests', 'daily_tokens', 'monthly_tokens',
  'session_requests', 'session_window_seconds'
}) do
  if redis.call('HEXISTS', KEYS[1], field) == 0 then
    return 'UNCONFIGURED'
  end
end
redis.call('HSET', KEYS[1], 'enabled', ARGV[1])
return 'OK'
"""

_RESERVE_SCRIPT = """
local values = redis.call('HMGET', KEYS[1],
  'enabled', 'daily_requests', 'monthly_requests', 'daily_tokens',
  'monthly_tokens', 'session_requests', 'session_window_seconds')
for index = 1, 7 do
  if not values[index] then return {'UNCONFIGURED'} end
end
if values[1] ~= '1' then return {'DISABLED'} end
if tonumber(values[7]) ~= tonumber(ARGV[6]) then return {'STALE_SETTINGS'} end
local estimated = tonumber(ARGV[1])
local current_day_requests = tonumber(redis.call('GET', KEYS[2]) or '0')
local current_month_requests = tonumber(redis.call('GET', KEYS[3]) or '0')
local current_day_tokens = tonumber(redis.call('GET', KEYS[4]) or '0')
local current_month_tokens = tonumber(redis.call('GET', KEYS[5]) or '0')
local current_session_requests = tonumber(redis.call('GET', KEYS[6]) or '0')
if current_day_requests + 1 > tonumber(values[2]) then return {'DAILY_REQUESTS'} end
if current_month_requests + 1 > tonumber(values[3]) then return {'MONTHLY_REQUESTS'} end
if current_day_tokens + estimated > tonumber(values[4]) then return {'DAILY_TOKENS'} end
if current_month_tokens + estimated > tonumber(values[5]) then return {'MONTHLY_TOKENS'} end
if current_session_requests + 1 > tonumber(values[6]) then return {'SESSION_REQUESTS'} end
redis.call('INCR', KEYS[2])
redis.call('INCR', KEYS[3])
redis.call('INCRBY', KEYS[4], estimated)
redis.call('INCRBY', KEYS[5], estimated)
redis.call('INCR', KEYS[6])
redis.call('EXPIREAT', KEYS[2], ARGV[2])
redis.call('EXPIREAT', KEYS[4], ARGV[2])
redis.call('EXPIREAT', KEYS[3], ARGV[3])
redis.call('EXPIREAT', KEYS[5], ARGV[3])
redis.call('EXPIREAT', KEYS[6], ARGV[4])
redis.call('HSET', KEYS[7],
  'state', 'reserved', 'reserved_tokens', estimated,
  'day_token_key', KEYS[4], 'month_token_key', KEYS[5])
redis.call('EXPIREAT', KEYS[7], ARGV[5])
return {'AUTHORIZED'}
"""

_RECONCILE_SCRIPT = """
if redis.call('HGET', KEYS[1], 'state') ~= 'reserved' then return 'IGNORED' end
local reserved = tonumber(redis.call('HGET', KEYS[1], 'reserved_tokens') or '0')
local day_key = redis.call('HGET', KEYS[1], 'day_token_key')
local month_key = redis.call('HGET', KEYS[1], 'month_token_key')
local actual = tonumber(ARGV[1])
local delta = actual - reserved
for _, key in ipairs({day_key, month_key}) do
  if redis.call('EXISTS', key) == 1 then
    local current = tonumber(redis.call('GET', key) or '0')
    local adjusted = current + delta
    if adjusted < 0 then adjusted = 0 end
    redis.call('SET', key, adjusted, 'KEEPTTL')
  end
end
redis.call('HSET', KEYS[1], 'state', 'reconciled', 'actual_tokens', actual)
return 'OK'
"""

_RELEASE_SCRIPT = """
if redis.call('HGET', KEYS[1], 'state') ~= 'reserved' then return 'IGNORED' end
local reserved = tonumber(redis.call('HGET', KEYS[1], 'reserved_tokens') or '0')
local day_key = redis.call('HGET', KEYS[1], 'day_token_key')
local month_key = redis.call('HGET', KEYS[1], 'month_token_key')
for _, key in ipairs({day_key, month_key}) do
  if redis.call('EXISTS', key) == 1 then
    local current = tonumber(redis.call('GET', key) or '0')
    local adjusted = current - reserved
    if adjusted < 0 then adjusted = 0 end
    redis.call('SET', key, adjusted, 'KEEPTTL')
  end
end
redis.call('HSET', KEYS[1], 'state', 'released')
return 'OK'
"""


class RedisQuotaBackend:
    """Enforce distributed hard quotas with Redis transactions and Lua.

    Parameters
    ----------
    redis_url
        Non-empty private Redis connection URL retained without connecting.
    key_prefix
        Namespace prefix whose hash tag keeps Lua keys in one Redis Cluster slot.
    client
        Optional injected Redis-compatible client for tests or managed lifecycles.

    Notes
    -----
    The default client is created lazily on the first backend operation. Each
    authorization revalidates settings and updates request, token, and session
    counters in one Lua operation. Raw session identifiers are never stored.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = "nlp-rag:{openai-quota}",
        client: Any | None = None,
    ) -> None:
        """Configure the Redis namespace and an optional injected client."""

        if not isinstance(redis_url, str) or not redis_url.strip():
            raise ValueError("redis_url must be a non-empty string")
        if not isinstance(key_prefix, str) or not key_prefix.strip():
            raise ValueError("key_prefix must be a non-empty string")
        self._redis_url = redis_url.strip()
        self._key_prefix = key_prefix.strip().rstrip(":")
        self._client = client

    def _redis(self) -> Any:
        if self._client is None:
            self._client = redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    @property
    def _settings_key(self) -> str:
        return f"{self._key_prefix}:limits"

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _current(now: datetime | None) -> datetime:
        current = datetime.now(UTC) if now is None else now
        if current.tzinfo is None:
            raise ValueError("quota timestamps must be timezone-aware")
        return current.astimezone(UTC)

    @staticmethod
    def _expiries(
        current: datetime, session_window_seconds: int
    ) -> tuple[int, int, int]:
        next_day = datetime.combine(current.date() + timedelta(days=1), time.min, UTC)
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1, tzinfo=UTC)
        else:
            next_month = datetime(current.year, current.month + 1, 1, tzinfo=UTC)
        day_expiry = int((next_day + timedelta(days=1)).timestamp())
        month_expiry = int((next_month + timedelta(days=1)).timestamp())
        window = int(current.timestamp()) // session_window_seconds
        session_expiry = (window + 1) * session_window_seconds + 60
        return day_expiry, month_expiry, session_expiry

    def _counter_keys(
        self, periods: contracts.QuotaPeriods, *, session_id: str
    ) -> tuple[str, str, str, str, str]:
        session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return (
            f"{self._key_prefix}:usage:requests:day:{periods.day}",
            f"{self._key_prefix}:usage:requests:month:{periods.month}",
            f"{self._key_prefix}:usage:tokens:day:{periods.day}",
            f"{self._key_prefix}:usage:tokens:month:{periods.month}",
            f"{self._key_prefix}:usage:session:{session_hash}:{periods.session_window}",
        )

    @staticmethod
    def _limits_from_mapping(values: dict[str, Any]) -> contracts.QuotaLimits:
        try:
            parsed = {name: int(values[name]) for name in _LIMIT_FIELDS}
        except (KeyError, TypeError, ValueError) as exc:
            raise contracts.QuotaUnavailableError(
                "OpenAI quota settings are missing or invalid in Redis."
            ) from exc
        try:
            return contracts.QuotaLimits(**parsed)
        except ValueError as exc:
            raise contracts.QuotaUnavailableError(
                "OpenAI quota settings are missing or invalid in Redis."
            ) from exc

    def inspect(self, *, now: datetime | None = None) -> contracts.QuotaUsageSnapshot:
        """Read owner-visible settings and aggregate active-period counters.

        Parameters
        ----------
        now
            Optional timezone-aware instant; defaults to current UTC time.

        Returns
        -------
        contracts.QuotaUsageSnapshot
            Settings plus day and month application counters; session counters are
            intentionally excluded from owner inspection.

        Raises
        ------
        ValueError
            If ``now`` is naive.
        contracts.QuotaUnavailableError
            If Redis is unavailable or stored settings or counters are invalid.
        """

        current = self._current(now)
        try:
            values = self._redis().hgetall(self._settings_key)
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend is unavailable."
            ) from exc
        normalized = {
            self._text(key): self._text(value) for key, value in values.items()
        }
        if not normalized:
            periods = contracts.QuotaPeriods.at(current, session_window_seconds=3600)
            return contracts.QuotaUsageSnapshot(
                enabled=None, limits=None, periods=periods
            )
        limits = self._limits_from_mapping(normalized)
        periods = contracts.QuotaPeriods.at(
            current, session_window_seconds=limits.session_window_seconds
        )
        keys = self._counter_keys(periods, session_id="owner-inspection")[:4]
        try:
            counters = self._redis().mget(list(keys))
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend is unavailable."
            ) from exc
        try:
            parsed_counters = [0 if value is None else int(value) for value in counters]
        except (TypeError, ValueError) as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota counters are invalid."
            ) from exc
        enabled_raw = normalized.get("enabled")
        if enabled_raw not in {"0", "1"}:
            raise contracts.QuotaUnavailableError(
                "OpenAI quota settings are missing or invalid in Redis."
            )
        return contracts.QuotaUsageSnapshot(
            enabled=enabled_raw == "1",
            limits=limits,
            periods=periods,
            daily_requests=parsed_counters[0],
            monthly_requests=parsed_counters[1],
            daily_tokens=parsed_counters[2],
            monthly_tokens=parsed_counters[3],
        )

    def set_limits(self, limits: contracts.QuotaLimits) -> None:
        """Replace all owner-controlled numeric limits transactionally.

        Parameters
        ----------
        limits
            Complete validated numeric limits.

        Raises
        ------
        contracts.QuotaUnavailableError
            If the Redis transaction cannot complete.

        Notes
        -----
        The transaction preserves an existing enable switch and initializes a
        missing switch to disabled.
        """

        mapping = {
            "daily_requests": limits.daily_requests,
            "monthly_requests": limits.monthly_requests,
            "daily_tokens": limits.daily_tokens,
            "monthly_tokens": limits.monthly_tokens,
            "session_requests": limits.session_requests,
            "session_window_seconds": limits.session_window_seconds,
        }
        try:
            pipeline = self._redis().pipeline(transaction=True)
            pipeline.hset(self._settings_key, mapping=mapping)
            pipeline.hsetnx(self._settings_key, "enabled", 0)
            pipeline.execute()
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend is unavailable."
            ) from exc

    def set_enabled(self, enabled: bool) -> None:
        """Immediately enable or disable OpenAI authorization in Redis.

        Parameters
        ----------
        enabled
            Whether subsequent OpenAI reservations may be authorized.

        Raises
        ------
        contracts.QuotaUnavailableError
            If Redis is unavailable or numeric limits are not fully configured.
        """

        try:
            result = self._redis().eval(
                _SET_ENABLED_SCRIPT,
                1,
                self._settings_key,
                "1" if enabled else "0",
            )
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend is unavailable."
            ) from exc
        if self._text(result) != "OK":
            raise contracts.QuotaUnavailableError(
                "OpenAI quota limits must be configured before changing availability."
            )

    def reserve(
        self,
        *,
        session_id: str,
        estimated_tokens: int,
        now: datetime | None = None,
    ) -> contracts.QuotaReservation:
        """Atomically authorize request counters and conservative token usage.

        Parameters
        ----------
        session_id
            Non-empty identifier hashed before use in a fixed-window counter key.
        estimated_tokens
            Positive conservative token charge for day and month counters.
        now
            Optional timezone-aware instant; defaults to current UTC time.

        Returns
        -------
        contracts.QuotaReservation
            Opaque reservation tied to the authorized UTC periods.

        Raises
        ------
        ValueError
            If inputs are invalid or ``now`` is naive.
        contracts.QuotaExhaustedError
            If disabled or a projected hard limit would be exceeded.
        contracts.QuotaUnavailableError
            If Redis cannot make a reliable authorization decision.

        Notes
        -----
        A preliminary inspection obtains the active window configuration; the Lua
        authorization then revalidates that configuration before atomically
        incrementing every applicable counter.
        """

        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be positive")
        snapshot = self.inspect(now=now)
        if snapshot.limits is None:
            raise contracts.QuotaUnavailableError(
                "OpenAI quota limits have not been configured."
            )
        current = self._current(now)
        periods = snapshot.periods
        day_expiry, month_expiry, session_expiry = self._expiries(
            current, snapshot.limits.session_window_seconds
        )
        counter_keys = self._counter_keys(periods, session_id=session_id)
        reservation_id = uuid.uuid4().hex
        reservation_key = f"{self._key_prefix}:reservation:{reservation_id}"
        try:
            raw_result = self._redis().eval(
                _RESERVE_SCRIPT,
                7,
                self._settings_key,
                *counter_keys,
                reservation_key,
                estimated_tokens,
                day_expiry,
                month_expiry,
                session_expiry,
                month_expiry,
                snapshot.limits.session_window_seconds,
            )
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend is unavailable."
            ) from exc
        result = self._text(
            raw_result[0] if isinstance(raw_result, list) else raw_result
        )
        reasons = {
            "DISABLED": "openai_disabled",
            "DAILY_REQUESTS": "daily_requests_exhausted",
            "MONTHLY_REQUESTS": "monthly_requests_exhausted",
            "DAILY_TOKENS": "daily_tokens_exhausted",
            "MONTHLY_TOKENS": "monthly_tokens_exhausted",
            "SESSION_REQUESTS": "session_requests_exhausted",
        }
        if result == "UNCONFIGURED":
            raise contracts.QuotaUnavailableError(
                "OpenAI quota limits have not been configured."
            )
        if result == "STALE_SETTINGS":
            raise contracts.QuotaUnavailableError(
                "OpenAI quota settings changed during authorization."
            )
        if result in reasons:
            raise contracts.QuotaExhaustedError(reasons[result])
        if result != "AUTHORIZED":
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend returned an invalid authorization result."
            )
        return contracts.QuotaReservation(
            reservation_id=reservation_id,
            reserved_tokens=estimated_tokens,
            periods=periods,
        )

    def _reservation_key(self, reservation: contracts.QuotaReservation) -> str:
        return f"{self._key_prefix}:reservation:{reservation.reservation_id}"

    def reconcile(
        self, reservation: contracts.QuotaReservation, *, actual_tokens: int
    ) -> None:
        """Reconcile conservative token counters with actual provider usage.

        Parameters
        ----------
        reservation
            Previously authorized opaque reservation.
        actual_tokens
            Non-negative provider-reported total token count.

        Raises
        ------
        ValueError
            If ``actual_tokens`` is negative.
        contracts.QuotaUnavailableError
            If Redis cannot perform the reconciliation.

        Notes
        -----
        The Lua transition is idempotent and request counters remain consumed.
        """

        if actual_tokens < 0:
            raise ValueError("actual_tokens must not be negative")
        try:
            self._redis().eval(
                _RECONCILE_SCRIPT,
                1,
                self._reservation_key(reservation),
                actual_tokens,
            )
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend could not reconcile OpenAI usage."
            ) from exc

    def release(self, reservation: contracts.QuotaReservation) -> None:
        """Safely release reserved tokens while retaining request counters.

        Parameters
        ----------
        reservation
            Previously authorized reservation known to be unbilled.

        Raises
        ------
        contracts.QuotaUnavailableError
            If Redis cannot perform the release.

        Notes
        -----
        The Lua transition is idempotent and never decrements request counters.
        """

        try:
            self._redis().eval(
                _RELEASE_SCRIPT,
                1,
                self._reservation_key(reservation),
            )
        except RedisError as exc:
            raise contracts.QuotaUnavailableError(
                "The Redis quota backend could not release reserved token usage."
            ) from exc
