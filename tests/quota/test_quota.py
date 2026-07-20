from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from src import quota


NOW = datetime(2026, 7, 19, 23, 59, tzinfo=UTC)


def limits(**overrides):
    values = {
        "daily_requests": 3,
        "monthly_requests": 10,
        "daily_tokens": 100,
        "monthly_tokens": 500,
        "session_requests": 2,
        "session_window_seconds": 3600,
    }
    values.update(overrides)
    return quota.contracts.QuotaLimits(**values)


def configured_backend(**overrides):
    backend = quota.memory.InMemoryQuotaBackend()
    backend.set_limits(limits(**overrides))
    backend.set_enabled(True)
    return backend


def test_authorization_reserves_then_reconciles_actual_tokens():
    backend = configured_backend()
    reservation = backend.reserve(session_id="session", estimated_tokens=40, now=NOW)
    assert backend.inspect(now=NOW).daily_tokens == 40

    backend.reconcile(reservation, actual_tokens=17)
    backend.reconcile(reservation, actual_tokens=2)
    usage = backend.inspect(now=NOW)

    assert usage.daily_requests == 1
    assert usage.monthly_requests == 1
    assert usage.daily_tokens == 17
    assert usage.monthly_tokens == 17


def test_release_is_idempotent_never_negative_and_keeps_request_count():
    backend = configured_backend()
    reservation = backend.reserve(session_id="session", estimated_tokens=40, now=NOW)

    backend.release(reservation)
    backend.release(reservation)
    usage = backend.inspect(now=NOW)

    assert usage.daily_requests == 1
    assert usage.daily_tokens == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"daily_requests": 1}, "daily_requests_exhausted"),
        (
            {
                "daily_requests": 2,
                "monthly_requests": 2,
                "session_requests": 1,
            },
            "session_requests_exhausted",
        ),
        ({"daily_tokens": 30}, "daily_tokens_exhausted"),
    ],
)
def test_hard_limits_reject_projected_usage(overrides, reason):
    backend = configured_backend(**overrides)
    backend.reserve(session_id="session", estimated_tokens=20, now=NOW)

    with pytest.raises(quota.contracts.QuotaExhaustedError) as captured:
        backend.reserve(session_id="session", estimated_tokens=20, now=NOW)
    assert captured.value.reason == reason


def test_concurrent_authorization_is_atomic():
    backend = configured_backend(
        daily_requests=1,
        monthly_requests=1,
        session_requests=1,
    )

    def authorize(index):
        try:
            backend.reserve(session_id=f"session-{index}", estimated_tokens=1, now=NOW)
        except quota.contracts.QuotaExhaustedError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(authorize, range(8)))

    assert outcomes.count(True) == 1
    assert backend.inspect(now=NOW).daily_requests == 1


def test_disable_reenable_and_utc_rollover_are_immediate():
    backend = configured_backend()
    backend.reserve(session_id="session", estimated_tokens=10, now=NOW)
    backend.set_enabled(False)
    with pytest.raises(quota.contracts.QuotaExhaustedError) as captured:
        backend.reserve(session_id="other", estimated_tokens=10, now=NOW)
    assert captured.value.reason == "openai_disabled"

    backend.set_enabled(True)
    next_day = datetime(2026, 7, 20, 0, 1, tzinfo=UTC)
    backend.reserve(session_id="other", estimated_tokens=10, now=next_day)
    assert backend.inspect(now=next_day).daily_requests == 1
    assert backend.inspect(now=next_day).monthly_requests == 2


def test_session_request_throttle_resets_at_the_next_utc_window():
    backend = configured_backend(
        daily_requests=10,
        monthly_requests=10,
        session_requests=1,
        session_window_seconds=3600,
    )
    backend.reserve(session_id="session", estimated_tokens=1, now=NOW)
    with pytest.raises(quota.contracts.QuotaExhaustedError) as captured:
        backend.reserve(session_id="session", estimated_tokens=1, now=NOW)
    assert captured.value.reason == "session_requests_exhausted"

    next_window = datetime(2026, 7, 20, 0, 1, tzinfo=UTC)
    backend.reserve(session_id="session", estimated_tokens=1, now=next_window)
    assert backend.inspect(now=next_window).daily_requests == 1


def test_monthly_request_limit_spans_daily_rollovers_then_resets():
    backend = configured_backend(
        daily_requests=2,
        monthly_requests=2,
        session_requests=10,
    )
    backend.reserve(session_id="one", estimated_tokens=1, now=NOW)
    backend.reserve(
        session_id="two",
        estimated_tokens=1,
        now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(quota.contracts.QuotaExhaustedError) as captured:
        backend.reserve(
            session_id="three",
            estimated_tokens=1,
            now=datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
        )
    assert captured.value.reason == "monthly_requests_exhausted"

    next_month = datetime(2026, 8, 1, 0, 1, tzinfo=UTC)
    backend.reserve(session_id="three", estimated_tokens=1, now=next_month)
    assert backend.inspect(now=next_month).monthly_requests == 1


def test_monthly_token_limit_spans_daily_rollovers():
    backend = configured_backend(
        daily_tokens=100,
        monthly_tokens=150,
        session_requests=10,
    )
    backend.reserve(session_id="one", estimated_tokens=80, now=NOW)

    with pytest.raises(quota.contracts.QuotaExhaustedError) as captured:
        backend.reserve(
            session_id="two",
            estimated_tokens=80,
            now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        )
    assert captured.value.reason == "monthly_tokens_exhausted"


def test_unconfigured_backend_fails_closed():
    backend = quota.memory.InMemoryQuotaBackend()
    with pytest.raises(quota.contracts.QuotaUnavailableError):
        backend.reserve(session_id="session", estimated_tokens=10, now=NOW)
    with pytest.raises(quota.contracts.QuotaUnavailableError):
        backend.set_enabled(True)


class RecordingRedis:
    def __init__(self):
        self.eval_calls = []

    def hgetall(self, _key):
        return {
            "enabled": "1",
            "daily_requests": "3",
            "monthly_requests": "10",
            "daily_tokens": "100",
            "monthly_tokens": "500",
            "session_requests": "2",
            "session_window_seconds": "3600",
        }

    def mget(self, _keys):
        return [None, None, None, None]

    def eval(self, script, numkeys, *args):
        self.eval_calls.append((script, numkeys, args))
        if "AUTHORIZED" in script:
            return ["AUTHORIZED"]
        return "OK"


def test_redis_adapter_uses_one_atomic_authorization_and_hashes_session_id():
    client = RecordingRedis()
    backend = quota.redis.RedisQuotaBackend(
        "redis://not-contacted",
        key_prefix="test:{quota}",
        client=client,
    )

    reservation = backend.reserve(
        session_id="raw-session-id", estimated_tokens=30, now=NOW
    )
    backend.reconcile(reservation, actual_tokens=12)
    backend.release(reservation)

    reserve_call = client.eval_calls[0]
    assert reserve_call[1] == 7
    assert "raw-session-id" not in " ".join(map(str, reserve_call[2]))
    assert any("usage:session:" in str(value) for value in reserve_call[2])
    assert reservation.reserved_tokens == 30


def test_redis_adapter_passes_tls_url_to_standard_client(monkeypatch):
    client = RecordingRedis()
    calls = []

    def from_url(redis_url, **kwargs):
        calls.append((redis_url, kwargs))
        return client

    monkeypatch.setattr(quota.redis.redis, "from_url", from_url)
    backend = quota.redis.RedisQuotaBackend(
        "rediss://example.invalid:6380/0",
        key_prefix="test:{quota}",
    )

    snapshot = backend.inspect(now=NOW)

    assert snapshot.enabled is True
    assert calls == [("rediss://example.invalid:6380/0", {"decode_responses": True})]


def test_redis_adapter_fails_closed_for_corrupt_counters():
    class CorruptCounterRedis(RecordingRedis):
        def mget(self, _keys):
            return ["not-an-integer", None, None, None]

    backend = quota.redis.RedisQuotaBackend(
        "redis://not-contacted",
        client=CorruptCounterRedis(),
    )

    with pytest.raises(quota.contracts.QuotaUnavailableError):
        backend.inspect(now=NOW)


def test_redis_adapter_fails_closed_when_limits_change_during_authorization():
    class StaleSettingsRedis(RecordingRedis):
        def eval(self, script, numkeys, *args):
            if "AUTHORIZED" in script:
                return ["STALE_SETTINGS"]
            return super().eval(script, numkeys, *args)

    backend = quota.redis.RedisQuotaBackend(
        "redis://not-contacted",
        client=StaleSettingsRedis(),
    )

    with pytest.raises(quota.contracts.QuotaUnavailableError):
        backend.reserve(session_id="session", estimated_tokens=1, now=NOW)
