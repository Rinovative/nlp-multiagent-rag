"""Generation-provider contracts, implementations, and routing.

Provides:
- contracts: immutable requests, results, usage, and project errors.
- huggingface: hosted open-model generation.
- openai: optional OpenAI generation behind router-owned quota enforcement.
- router: deterministic provider selection and fallback.
"""

from __future__ import annotations

from . import providers_contracts as contracts
from . import providers_generation_huggingface as huggingface
from . import providers_generation_openai as openai
from . import providers_router as router

__all__ = ["contracts", "huggingface", "openai", "router"]
