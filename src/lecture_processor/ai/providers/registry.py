from .base import ProviderRequestError
from .gemini import GeminiProvider
from .mock import MockProvider


def build_provider(name: str, *, api_key: str = "", model: str = ""):
    normalized = (name or "none").lower()
    if normalized == "mock":
        return MockProvider(api_key=api_key, model=model or MockProvider.info().default_model)
    if normalized == "gemini":
        return GeminiProvider(api_key=api_key, model=model or GeminiProvider.info().default_model)
    raise ProviderRequestError(f"Unsupported AI provider: {name}")


def provider_info(name: str):
    normalized = (name or "none").lower()
    if normalized == "mock":
        return MockProvider.info()
    if normalized == "gemini":
        return GeminiProvider.info()
    raise ProviderRequestError(f"Unsupported AI provider: {name}")
