import pytest

from src import configuration

AppConfig = configuration.runtime.AppConfig
ConfigurationError = configuration.runtime.ConfigurationError


def test_streamlit_boundary_values_override_environment():
    config = AppConfig.from_sources(
        secrets={
            "HUGGINGFACE_API_TOKEN": "secret-token",
            "GENERATION_PROVIDER": "huggingface",
            "MAX_UPLOAD_MB": "7",
        },
        environ={
            "HUGGINGFACE_API_TOKEN": "environment-token",
            "GENERATION_PROVIDER": "openai",
            "MAX_UPLOAD_MB": "3",
        },
    )

    assert config.huggingface_api_token == "secret-token"
    assert config.generation_provider == "huggingface"
    assert config.max_upload_mb == 7


def test_credential_free_defaults_use_local_e5_and_auto_free_route():
    config = AppConfig.from_sources(secrets={}, environ={})

    assert config.embedding_model == "intfloat/multilingual-e5-small"
    assert config.embedding_dimension == 384
    assert config.embedding_uses_e5_prefixes is True
    assert config.generation_provider == "auto"
    assert config.openai_api_key is None
    assert config.redis_url is None
    with pytest.raises(ConfigurationError, match="HUGGINGFACE_API_TOKEN"):
        config.require_huggingface_token()
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        config.require_openai_key()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EMBEDDING_DIMENSION", "not-an-integer"),
        ("MAX_OUTPUT_TOKENS", "0"),
        ("PROVIDER_TIMEOUT_SECONDS", "nope"),
        ("GENERATION_PROVIDER", "unknown"),
        ("OPENAI_FALLBACK_ENABLED", "sometimes"),
    ],
)
def test_invalid_configuration_is_rejected_with_canonical_variable(name, value):
    with pytest.raises(ConfigurationError, match=name):
        AppConfig.from_sources(secrets={}, environ={name: value})


@pytest.mark.parametrize(("raw", "expected"), [("yes", True), ("0", False)])
def test_boolean_configuration_is_parsed_strictly(raw, expected):
    config = AppConfig.from_sources(
        secrets={}, environ={"OPENAI_FALLBACK_ENABLED": raw}
    )
    assert config.openai_fallback_enabled is expected
