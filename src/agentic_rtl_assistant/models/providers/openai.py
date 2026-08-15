from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI, OpenAIError

from agentic_rtl_assistant.models.types import (
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60,
        retries: int = 1,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=retries,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": [
                {
                    "role": "developer" if message.role == "system" else message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        try:
            response = await self._client.responses.create(**payload)
        except OpenAIError as exc:
            raise ModelProviderError(f"OpenAI request failed: {exc}") from exc

        usage = response.usage
        cached = None
        if usage is not None and usage.input_tokens_details is not None:
            cached = getattr(usage.input_tokens_details, "cached_tokens", None)
        normalized = TokenUsage(
            input_tokens=usage.input_tokens if usage is not None else 0,
            output_tokens=usage.output_tokens if usage is not None else 0,
            cached_input_tokens=cached,
            llm_calls=1,
        )
        return ModelResponse(
            content=response.output_text,
            provider=self.name,
            model=request.model,
            usage=normalized,
            metadata={"response_id": response.id},
        )
