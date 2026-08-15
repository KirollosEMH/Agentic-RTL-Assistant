from __future__ import annotations

import os
from collections.abc import Mapping

from agentic_rtl_assistant.config.models import ModelProfile, ModelsSettings, ProviderSettings
from agentic_rtl_assistant.models.base import ModelProvider
from agentic_rtl_assistant.models.providers import (
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


class ModelProviderFactory:
    def __init__(
        self,
        settings: ModelsSettings,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._environment = environment or os.environ

    def create_for_profile(self, profile_name: str) -> tuple[ModelProvider, ModelProfile]:
        try:
            profile = self._settings.profiles[profile_name]
            provider = self._settings.providers[profile.provider]
        except KeyError as exc:
            raise ValueError(f"unknown model profile or provider: {profile_name}") from exc
        return self._create(profile, provider), profile

    def _secret(self, variable: str | None, *, required: bool) -> str:
        value = self._environment.get(variable, "") if variable else ""
        if required and not value:
            raise ValueError(f"required credential environment variable is unset: {variable}")
        return value

    def _create(self, profile: ModelProfile, provider: ProviderSettings) -> ModelProvider:
        name = profile.provider
        if name == "openai":
            return OpenAIProvider(
                api_key=self._secret(provider.api_key_env, required=True),
                base_url=provider.base_url,
                timeout=profile.timeout_seconds,
                retries=profile.retries,
            )

        classes = {
            "ollama": OllamaProvider,
            "openrouter": OpenRouterProvider,
            "groq": GroqProvider,
        }
        try:
            provider_class = classes[name]
        except KeyError as exc:
            raise ValueError(f"unsupported model provider: {name}") from exc
        base_url = provider.base_url
        if not base_url:
            raise ValueError(f"provider {name} requires a base_url")
        api_key = self._secret(provider.api_key_env, required=True)
        return provider_class(
            api_key=api_key,
            base_url=base_url,
            timeout=profile.timeout_seconds,
            retries=profile.retries,
        )
