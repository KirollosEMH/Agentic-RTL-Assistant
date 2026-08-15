from pathlib import Path

import pytest
from openai import AsyncOpenAI

from agentic_rtl_assistant.config import load_config
from agentic_rtl_assistant.models.factory import ModelProviderFactory
from agentic_rtl_assistant.models.providers.ollama import OllamaProvider
from agentic_rtl_assistant.models.providers.openrouter import OpenRouterProvider
from agentic_rtl_assistant.models.types import ModelRequest, TokenUsage
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.tokens import aggregate_usage
from agentic_rtl_assistant.telemetry.traces import EventType, ExecutionTrace


def test_provider_factory_builds_configured_ollama_provider(repository_root: Path) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})

    provider, profile = ModelProviderFactory(
        config.models, {"OLLAMA_API_KEY": "test-key"}
    ).create_for_profile("fast")

    assert isinstance(provider, OllamaProvider)
    assert isinstance(provider._client, AsyncOpenAI)
    assert str(provider._client.base_url) == "https://ollama.com/v1/"
    assert profile.model == "gpt-oss:120b"


def test_provider_factory_requires_ollama_api_key(repository_root: Path) -> None:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})

    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        ModelProviderFactory(config.models, {}).create_for_profile("fast")


def test_token_aggregation_preserves_unknown_cached_usage() -> None:
    usage = aggregate_usage(
        [
            TokenUsage(10, 2, 4, 1),
            TokenUsage(5, 1, None, 1),
        ]
    )

    assert usage.input_tokens == 15
    assert usage.output_tokens == 3
    assert usage.cached_input_tokens is None
    assert usage.llm_calls == 2


def test_openrouter_forwards_session_id_for_sticky_routing() -> None:
    provider = OpenRouterProvider(
        api_key="test-key", base_url="https://openrouter.ai/api/v1"
    )
    request = ModelRequest(
        model="test/model",
        messages=(),
        metadata={"session_id": "session-123"},
    )

    assert provider._extra_body(request) == {"session_id": "session-123"}


def test_telemetry_collects_structured_events() -> None:
    collector = TelemetryCollector()
    event = ExecutionTrace("request", "test", EventType.TOOL_STARTED)

    collector.record(event)

    assert collector.for_request("request") == (event,)
