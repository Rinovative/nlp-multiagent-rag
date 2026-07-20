import logging
from datetime import UTC, datetime

import pytest

from src import providers, quota


NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def request():
    return providers.contracts.GenerationRequest(
        messages=(
            providers.contracts.GenerationMessage(role="user", content="Question"),
        ),
        max_output_tokens=20,
        estimated_input_tokens=10,
    )


class FakeProvider:
    def __init__(self, provider_id, *, error=None, usage=None):
        self.provider_id = provider_id
        self.model_id = f"{provider_id}-model"
        self.error = error
        self.usage = usage or providers.contracts.GenerationUsage(4, 3)
        self.calls = 0

    def generate(self, generation_request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return providers.contracts.GenerationResult(
            answer=f"{self.provider_id} answer",
            provider_id=self.provider_id,
            model_id=self.model_id,
            usage=self.usage,
        )


class FixedClockQuota:
    def __init__(self, backend):
        self.backend = backend
        self.reserve_calls = 0

    def inspect(self, *, now=None):
        return self.backend.inspect(now=NOW)

    def set_limits(self, limits):
        self.backend.set_limits(limits)

    def set_enabled(self, enabled):
        self.backend.set_enabled(enabled)

    def reserve(self, *, session_id, estimated_tokens, now=None):
        self.reserve_calls += 1
        return self.backend.reserve(
            session_id=session_id, estimated_tokens=estimated_tokens, now=NOW
        )

    def reconcile(self, reservation, *, actual_tokens):
        self.backend.reconcile(reservation, actual_tokens=actual_tokens)

    def release(self, reservation):
        self.backend.release(reservation)


class UnavailableQuota:
    def reserve(self, **_kwargs):
        raise quota.contracts.QuotaUnavailableError("unavailable")


class ExhaustedQuota:
    def __init__(self, reason):
        self.reason = reason
        self.reserve_calls = 0

    def reserve(self, **_kwargs):
        self.reserve_calls += 1
        raise quota.contracts.QuotaExhaustedError(self.reason)


def configured_quota(*, daily_tokens=1000):
    backend = quota.memory.InMemoryQuotaBackend()
    backend.set_limits(
        quota.contracts.QuotaLimits(
            daily_requests=10,
            monthly_requests=100,
            daily_tokens=daily_tokens,
            monthly_tokens=10_000,
            session_requests=5,
            session_window_seconds=3600,
        )
    )
    backend.set_enabled(True)
    return FixedClockQuota(backend)


def test_huggingface_mode_never_touches_openai_or_quota():
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    hard_quota = configured_quota()
    router = providers.router.GenerationRouter(
        mode="huggingface",
        free_provider=free,
        openai_provider=openai,
        quota_backend=hard_quota,
    )

    result = router.generate(request(), session_id="session")

    assert result.provider_id == "huggingface"
    assert result.fallback_occurred is False
    assert free.calls == 1
    assert openai.calls == 0
    assert hard_quota.reserve_calls == 0


def test_auto_without_openai_configuration_uses_free_provider_without_fallback():
    free = FakeProvider("huggingface")
    router = providers.router.GenerationRouter(
        mode="auto", free_provider=free, openai_provider=None
    )

    result = router.generate(request(), session_id="session")

    assert result.provider_id == "huggingface"
    assert result.fallback_occurred is False


def test_auto_falls_back_when_redis_quota_is_unavailable():
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=None,
    )

    result = router.generate(request(), session_id="session")

    assert result.provider_id == "huggingface"
    assert result.fallback_reason == "openai_quota_unavailable"
    assert openai.calls == 0
    assert free.calls == 1


def test_auto_falls_back_when_configured_redis_backend_fails():
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=UnavailableQuota(),
    )

    result = router.generate(request(), session_id="session")

    assert result.fallback_reason == "openai_quota_unavailable"
    assert openai.calls == 0
    assert free.calls == 1


def test_explicit_openai_without_redis_fails_closed(caplog):
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    router = providers.router.GenerationRouter(
        mode="openai",
        free_provider=free,
        openai_provider=openai,
        quota_backend=None,
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(quota.contracts.QuotaUnavailableError):
            router.generate(request(), session_id="session")

    assert openai.calls == 0
    assert free.calls == 0
    assert (
        "openai_quota_authorization_denied provider=openai model=openai-model "
        "fallback_reason=openai_quota_unavailable "
        "provider_call_attempted=false"
    ) in caplog.text


def test_openai_success_is_authorized_and_reconciled_to_actual_usage():
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    hard_quota = configured_quota()
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=hard_quota,
    )

    result = router.generate(request(), session_id="session")
    usage = hard_quota.inspect(now=NOW)

    assert result.provider_id == "openai"
    assert result.fallback_occurred is False
    assert usage.daily_requests == 1
    assert usage.daily_tokens == 7
    assert free.calls == 0


def test_auto_uses_openai_five_times_then_huggingface_for_session_limit(caplog):
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    hard_quota = configured_quota()
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=hard_quota,
    )

    with caplog.at_level(logging.INFO):
        results = [
            router.generate(request(), session_id="one-browser-session")
            for _ in range(6)
        ]

    assert [result.provider_id for result in results[:5]] == ["openai"] * 5
    assert all(result.fallback_occurred is False for result in results[:5])
    assert results[5].provider_id == "huggingface"
    assert results[5].model_id == "huggingface-model"
    assert results[5].fallback_occurred is True
    assert results[5].fallback_reason == "session_requests_exhausted"
    assert openai.calls == 5
    assert free.calls == 1
    assert hard_quota.reserve_calls == 6
    assert (
        "openai_quota_authorization_denied provider=openai model=openai-model "
        "fallback_reason=session_requests_exhausted "
        "provider_call_attempted=false"
    ) in caplog.text
    assert (
        "generation_route_selected provider=huggingface model=huggingface-model "
        "fallback_reason=session_requests_exhausted "
        "provider_call_attempted=false"
    ) in caplog.text


@pytest.mark.parametrize(
    "reason",
    [
        "daily_requests_exhausted",
        "monthly_requests_exhausted",
        "daily_tokens_exhausted",
        "monthly_tokens_exhausted",
        "openai_disabled",
    ],
)
def test_auto_uses_huggingface_for_each_global_quota_denial(reason):
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    denied_quota = ExhaustedQuota(reason)
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=denied_quota,
    )

    result = router.generate(request(), session_id="session")

    assert result.provider_id == "huggingface"
    assert result.model_id == "huggingface-model"
    assert result.fallback_occurred is True
    assert result.fallback_reason == reason
    assert denied_quota.reserve_calls == 1
    assert openai.calls == 0
    assert free.calls == 1


def test_auto_falls_back_once_when_quota_is_exhausted():
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    hard_quota = configured_quota(daily_tokens=25)
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=hard_quota,
    )

    result = router.generate(request(), session_id="session")

    assert result.provider_id == "huggingface"
    assert result.fallback_occurred is True
    assert result.fallback_reason == "daily_tokens_exhausted"
    assert openai.calls == 0
    assert free.calls == 1


@pytest.mark.parametrize(
    ("error", "reason", "expected_reserved_tokens"),
    [
        (
            providers.contracts.GenerationRateLimitError("rate"),
            "openai_rate_limited",
            0,
        ),
        (
            providers.contracts.GenerationTemporaryError("temporary"),
            "openai_temporarily_unavailable",
            30,
        ),
    ],
)
def test_auto_falls_back_once_for_classified_openai_capacity_failures(
    error, reason, expected_reserved_tokens
):
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai", error=error)
    hard_quota = configured_quota()
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=hard_quota,
    )

    result = router.generate(request(), session_id="session")

    assert result.fallback_reason == reason
    assert result.provider_id == "huggingface"
    assert openai.calls == 1
    assert free.calls == 1
    assert hard_quota.inspect(now=NOW).daily_tokens == expected_reserved_tokens


def test_malformed_openai_response_keeps_conservative_reservation():
    free = FakeProvider("huggingface")
    openai = FakeProvider(
        "openai",
        error=providers.contracts.GenerationResponseError("malformed"),
    )
    hard_quota = configured_quota()
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=hard_quota,
    )

    with pytest.raises(providers.contracts.GenerationResponseError):
        router.generate(request(), session_id="session")

    assert free.calls == 0
    assert hard_quota.inspect(now=NOW).daily_tokens == 30


def test_failed_fallback_is_not_retried_or_routed_recursively():
    free = FakeProvider(
        "huggingface",
        error=providers.contracts.GenerationTemporaryError("free unavailable"),
    )
    openai = FakeProvider(
        "openai",
        error=providers.contracts.GenerationTemporaryError("openai unavailable"),
    )
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=configured_quota(),
    )

    with pytest.raises(providers.contracts.GenerationFallbackError) as captured:
        router.generate(request(), session_id="session")

    assert captured.value.provider_id == "huggingface"
    assert captured.value.model_id == "huggingface-model"
    assert captured.value.fallback_reason == "openai_temporarily_unavailable"
    assert captured.value.provider_error_category == "temporary"
    assert isinstance(
        captured.value.__cause__, providers.contracts.GenerationTemporaryError
    )
    assert openai.calls == 1
    assert free.calls == 1


def test_explicit_openai_reports_missing_provider_configuration():
    router = providers.router.GenerationRouter(
        mode="openai",
        free_provider=FakeProvider("huggingface"),
        openai_provider=None,
        quota_backend=configured_quota(),
    )

    with pytest.raises(providers.contracts.GenerationConfigurationError):
        router.generate(request(), session_id="session")


@pytest.mark.parametrize(
    "error",
    [
        providers.contracts.GenerationAuthenticationError("invalid"),
        providers.contracts.GenerationInvalidRequestError("invalid"),
        providers.contracts.GenerationSafetyError("unsafe"),
        providers.contracts.GenerationResponseError("malformed"),
    ],
)
def test_non_capacity_openai_failures_never_fall_back(error):
    free = FakeProvider("huggingface")
    openai = FakeProvider(
        "openai",
        error=error,
    )
    router = providers.router.GenerationRouter(
        mode="auto",
        free_provider=free,
        openai_provider=openai,
        quota_backend=configured_quota(),
    )

    with pytest.raises(type(error)):
        router.generate(request(), session_id="session")
    assert free.calls == 0


def test_explicit_openai_mode_fails_closed_unless_fallback_is_enabled():
    free = FakeProvider("huggingface")
    openai = FakeProvider("openai")
    disabled_quota = configured_quota()
    disabled_quota.set_enabled(False)
    strict_router = providers.router.GenerationRouter(
        mode="openai",
        free_provider=free,
        openai_provider=openai,
        quota_backend=disabled_quota,
    )
    with pytest.raises(quota.contracts.QuotaExhaustedError):
        strict_router.generate(request(), session_id="session")

    fallback_router = providers.router.GenerationRouter(
        mode="openai",
        free_provider=free,
        openai_provider=openai,
        quota_backend=disabled_quota,
        openai_fallback_enabled=True,
    )
    result = fallback_router.generate(request(), session_id="session")
    assert result.provider_id == "huggingface"
    assert result.fallback_reason == "openai_disabled"
