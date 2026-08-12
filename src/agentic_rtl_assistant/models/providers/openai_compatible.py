"""Adapter for providers exposing OpenAI-compatible chat completions."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAIError

from agentic_rtl_assistant.models.types import (
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
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
        kwargs: dict[str, object] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
        }
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise ModelProviderError(f"{self.name} request failed: {exc}") from exc
        usage = response.usage
        cached = None
        if usage is not None and usage.prompt_tokens_details is not None:
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", None)
        content = response.choices[0].message.content or ""
        return ModelResponse(
            content=content,
            provider=self.name,
            model=request.model,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage is not None else 0,
                output_tokens=usage.completion_tokens if usage is not None else 0,
                cached_input_tokens=cached,
                llm_calls=1,
            ),
            metadata={"response_id": response.id},
        )
