"""
===============================================================================
configuration_runtime.py
===============================================================================
Resolve and validate immutable application settings.

Responsibilities:
  - Combine Streamlit-provided values with environment variables.
  - Validate provider modes, model identifiers, and resource limits.
  - Expose feature-specific credential checks without creating clients.

Design principles:
  - Resolve each setting once through a typed immutable contract.
  - Prefer Streamlit secrets over matching environment values.

Boundaries:
  - Does not import Streamlit, connect to Redis, or load models.
  - Does not test whether configured external credentials are live.
===============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, cast

__all__ = ["AppConfig", "ConfigurationError", "GenerationMode"]

# Stable provider-selection values accepted by application configuration.
GenerationMode = Literal["auto", "huggingface", "openai"]


class ConfigurationError(RuntimeError):
    """Represent an invalid or missing setting safe for the UI boundary."""


@dataclass(frozen=True)
class AppConfig:
    """Hold canonical immutable settings for one application process.

    Parameters
    ----------
    generation_provider
        ``auto``, ``huggingface``, or ``openai`` routing mode.
    huggingface_api_token
        Optional owner token required only when Hugging Face generation runs.
    huggingface_generation_model
        Hosted Hugging Face model identifier.
    openai_api_key
        Optional owner key required only when OpenAI generation is selected.
    openai_generation_model
        Optional OpenAI model identifier.
    openai_fallback_enabled
        Whether explicit OpenAI mode permits supported free-provider fallback.
    redis_url
        Optional private Redis URL required for quota-controlled OpenAI calls.
    quota_key_prefix
        Redis namespace with a cluster hash tag for atomic quota scripts.
    embedding_model
        Local SentenceTransformer model identifier used for ingestion and queries.
    embedding_dimension
        Fixed vector dimension expected from the local embedding model.
    embedding_batch_size
        Positive number of document passages per local embedding batch.
    max_upload_mb
        Positive per-file and combined upload bound in mebibytes.
    max_input_characters
        Positive character bound for assembled generation input.
    max_output_tokens
        Positive provider completion-token bound.
    max_history_messages
        Positive number of messages retained per session.
    retrieval_top_k
        Positive maximum number of FAISS records retrieved per question.
    provider_timeout_seconds
        Positive hosted-provider timeout in seconds.

    Notes
    -----
    The frozen dataclass validates direct construction as well as values resolved
    by :meth:`from_sources`. Credentials are stored but never verified remotely.
    """

    generation_provider: GenerationMode = "auto"
    huggingface_api_token: str | None = None
    huggingface_generation_model: str = "Qwen/Qwen2.5-7B-Instruct"
    openai_api_key: str | None = None
    openai_generation_model: str = "gpt-4o-mini"
    openai_fallback_enabled: bool = False
    redis_url: str | None = None
    quota_key_prefix: str = "nlp-rag:{openai-quota}"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    max_upload_mb: int = 20
    max_input_characters: int = 24_000
    max_output_tokens: int = 512
    max_history_messages: int = 10
    retrieval_top_k: int = 5
    provider_timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        """Reject invalid direct construction as well as invalid source values."""

        if self.generation_provider not in {"auto", "huggingface", "openai"}:
            raise ConfigurationError(
                "GENERATION_PROVIDER must be auto, huggingface, or openai."
            )
        for name, value in (
            ("HUGGINGFACE_GENERATION_MODEL", self.huggingface_generation_model),
            ("OPENAI_GENERATION_MODEL", self.openai_generation_model),
            ("EMBEDDING_MODEL", self.embedding_model),
            ("OPENAI_QUOTA_KEY_PREFIX", self.quota_key_prefix),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty string.")
        for integer_name, integer_value in (
            ("EMBEDDING_DIMENSION", self.embedding_dimension),
            ("EMBEDDING_BATCH_SIZE", self.embedding_batch_size),
            ("MAX_UPLOAD_MB", self.max_upload_mb),
            ("MAX_INPUT_CHARACTERS", self.max_input_characters),
            ("MAX_OUTPUT_TOKENS", self.max_output_tokens),
            ("MAX_HISTORY_MESSAGES", self.max_history_messages),
            ("RETRIEVAL_TOP_K", self.retrieval_top_k),
        ):
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value <= 0
            ):
                raise ConfigurationError(f"{integer_name} must be a positive integer.")
        if self.provider_timeout_seconds <= 0:
            raise ConfigurationError("PROVIDER_TIMEOUT_SECONDS must be positive.")

    @classmethod
    def from_sources(
        cls,
        *,
        secrets: Mapping[str, object] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "AppConfig":
        """Resolve secrets first, then environment values, then safe defaults.

        Parameters
        ----------
        secrets
            Flat mapping supplied by the Streamlit application boundary.
        environ
            Environment mapping. Defaults to :data:`os.environ`.

        Returns
        -------
        AppConfig
            Validated immutable settings with secret values taking precedence.

        Raises
        ------
        ConfigurationError
            If a configured value cannot be parsed or violates its allowed range.
        """

        environment = os.environ if environ is None else environ
        secret_values = {} if secrets is None else secrets
        defaults = cls()

        def value(name: str, default: str | None = None) -> str | None:
            secret_value = secret_values.get(name)
            if secret_value is not None and str(secret_value).strip():
                return str(secret_value).strip()
            environment_value = environment.get(name)
            if environment_value is not None and environment_value.strip():
                return environment_value.strip()
            return default

        def integer(name: str, default: int) -> int:
            raw_value = value(name, str(default))
            try:
                parsed = int(cast(str, raw_value))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"{name} must be an integer.") from exc
            if parsed <= 0:
                raise ConfigurationError(f"{name} must be greater than zero.")
            return parsed

        def number(name: str, default: float) -> float:
            raw_value = value(name, str(default))
            try:
                parsed = float(cast(str, raw_value))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(f"{name} must be a number.") from exc
            if parsed <= 0:
                raise ConfigurationError(f"{name} must be greater than zero.")
            return parsed

        def boolean(name: str, default: bool) -> bool:
            raw_value = cast(str, value(name, str(default))).strip().lower()
            if raw_value in {"1", "true", "yes", "on"}:
                return True
            if raw_value in {"0", "false", "no", "off"}:
                return False
            raise ConfigurationError(
                f"{name} must be one of true, false, 1, 0, yes, no, on, or off."
            )

        mode = cast(
            str, value("GENERATION_PROVIDER", defaults.generation_provider)
        ).lower()
        if mode not in {"auto", "huggingface", "openai"}:
            raise ConfigurationError(
                "GENERATION_PROVIDER must be auto, huggingface, or openai."
            )

        return cls(
            generation_provider=cast(GenerationMode, mode),
            huggingface_api_token=value("HUGGINGFACE_API_TOKEN"),
            huggingface_generation_model=cast(
                str,
                value(
                    "HUGGINGFACE_GENERATION_MODEL",
                    defaults.huggingface_generation_model,
                ),
            ),
            openai_api_key=value("OPENAI_API_KEY"),
            openai_generation_model=cast(
                str,
                value(
                    "OPENAI_GENERATION_MODEL",
                    defaults.openai_generation_model,
                ),
            ),
            openai_fallback_enabled=boolean(
                "OPENAI_FALLBACK_ENABLED", defaults.openai_fallback_enabled
            ),
            redis_url=value("REDIS_URL"),
            quota_key_prefix=cast(
                str,
                value("OPENAI_QUOTA_KEY_PREFIX", defaults.quota_key_prefix),
            ),
            embedding_model=cast(
                str, value("EMBEDDING_MODEL", defaults.embedding_model)
            ),
            embedding_dimension=integer(
                "EMBEDDING_DIMENSION", defaults.embedding_dimension
            ),
            embedding_batch_size=integer(
                "EMBEDDING_BATCH_SIZE", defaults.embedding_batch_size
            ),
            max_upload_mb=integer("MAX_UPLOAD_MB", defaults.max_upload_mb),
            max_input_characters=integer(
                "MAX_INPUT_CHARACTERS", defaults.max_input_characters
            ),
            max_output_tokens=integer("MAX_OUTPUT_TOKENS", defaults.max_output_tokens),
            max_history_messages=integer(
                "MAX_HISTORY_MESSAGES", defaults.max_history_messages
            ),
            retrieval_top_k=integer("RETRIEVAL_TOP_K", defaults.retrieval_top_k),
            provider_timeout_seconds=number(
                "PROVIDER_TIMEOUT_SECONDS", defaults.provider_timeout_seconds
            ),
        )

    @property
    def openai_is_configured(self) -> bool:
        """Return whether optional OpenAI generation has an owner key."""

        return self.openai_api_key is not None

    @property
    def embedding_uses_e5_prefixes(self) -> bool:
        """Return whether the configured embedding family requires E5 prefixes."""

        return "e5" in self.embedding_model.casefold()

    def require_huggingface_token(self) -> str:
        """Return the hosted-inference token required at generation time.

        Returns
        -------
        str
            Configured owner token.

        Raises
        ------
        ConfigurationError
            If no Hugging Face token is configured.
        """

        if self.huggingface_api_token is None:
            raise ConfigurationError(
                "HUGGINGFACE_API_TOKEN is required when answer generation is invoked."
            )
        return self.huggingface_api_token

    def require_openai_key(self) -> str:
        """Return the owner OpenAI key required at client creation time.

        Returns
        -------
        str
            Configured owner key.

        Raises
        ------
        ConfigurationError
            If no OpenAI key is configured.
        """

        if self.openai_api_key is None:
            raise ConfigurationError(
                "OPENAI_API_KEY is required when OpenAI generation is selected."
            )
        return self.openai_api_key

    def require_redis_url(self) -> str:
        """Return the private Redis URL required for paid-provider authorization.

        Returns
        -------
        str
            Configured Redis connection URL.

        Raises
        ------
        ConfigurationError
            If no Redis URL is configured, preserving fail-closed OpenAI routing.
        """

        if self.redis_url is None:
            raise ConfigurationError(
                "REDIS_URL is required for quota-controlled OpenAI generation."
            )
        return self.redis_url
