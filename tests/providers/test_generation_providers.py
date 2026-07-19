from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError
from openai import AuthenticationError, BadRequestError, RateLimitError

from src import providers


def request():
    return providers.contracts.GenerationRequest(
        messages=(
            providers.contracts.GenerationMessage(role="user", content="Question"),
        ),
        max_output_tokens=25,
        estimated_input_tokens=12,
    )


class FakeHuggingFaceClient:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  HF answer  "))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        )


def hf_http_error(status_code):
    return HfHubHTTPError(
        "provider detail with secret-value",
        response=SimpleNamespace(status_code=status_code, headers={}, request=None),
    )


def test_huggingface_provider_uses_supported_chat_shape_and_normalizes_usage():
    client = FakeHuggingFaceClient()
    provider = providers.huggingface.HuggingFaceGenerationProvider(
        lambda: client, model_id="Qwen/Qwen2.5-7B-Instruct"
    )

    result = provider.generate(request())

    assert client.calls == [
        {
            "messages": [{"role": "user", "content": "Question"}],
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "max_tokens": 25,
            "temperature": 0.2,
        }
    ]
    assert result.answer == "HF answer"
    assert result.provider_id == "huggingface"
    assert result.usage.total_tokens == 14


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (hf_http_error(401), providers.contracts.GenerationAuthenticationError),
        (hf_http_error(402), providers.contracts.GenerationRateLimitError),
        (hf_http_error(429), providers.contracts.GenerationRateLimitError),
        (hf_http_error(503), providers.contracts.GenerationTemporaryError),
        (hf_http_error(400), providers.contracts.GenerationInvalidRequestError),
        (
            InferenceTimeoutError("secret-value"),
            providers.contracts.GenerationTemporaryError,
        ),
    ],
)
def test_huggingface_sdk_errors_are_classified_without_leaking_detail(
    error, expected_type
):
    provider = providers.huggingface.HuggingFaceGenerationProvider(
        lambda: FakeHuggingFaceClient(error=error), model_id="model"
    )
    with pytest.raises(expected_type) as captured:
        provider.generate(request())
    assert "secret-value" not in str(captured.value)


class FakeOpenAICompletions:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=" OpenAI answer "))
            ],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4),
        )


class FakeOpenAIClient:
    def __init__(self, *, error=None):
        self.chat = SimpleNamespace(completions=FakeOpenAICompletions(error=error))


def openai_error(error_type, status_code, *, body=None):
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return error_type("secret-value", response=response, body=body)


def test_openai_provider_normalizes_actual_usage():
    client = FakeOpenAIClient()
    provider = providers.openai.OpenAIGenerationProvider(
        lambda: client, model_id="gpt-4o-mini"
    )

    result = provider.generate(request())

    assert result.answer == "OpenAI answer"
    assert result.provider_id == "openai"
    assert result.model_id == "gpt-4o-mini"
    assert result.usage == providers.contracts.GenerationUsage(9, 4)
    assert client.chat.completions.calls[0]["max_tokens"] == 25


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (
            openai_error(AuthenticationError, 401),
            providers.contracts.GenerationAuthenticationError,
        ),
        (
            openai_error(RateLimitError, 429),
            providers.contracts.GenerationRateLimitError,
        ),
        (
            openai_error(BadRequestError, 400),
            providers.contracts.GenerationInvalidRequestError,
        ),
        (
            openai_error(BadRequestError, 400, body={"code": "content_filter"}),
            providers.contracts.GenerationSafetyError,
        ),
    ],
)
def test_openai_sdk_errors_are_classified_without_leaking_detail(error, expected_type):
    provider = providers.openai.OpenAIGenerationProvider(
        lambda: FakeOpenAIClient(error=error), model_id="gpt-4o-mini"
    )
    with pytest.raises(expected_type) as captured:
        provider.generate(request())
    assert "secret-value" not in str(captured.value)
