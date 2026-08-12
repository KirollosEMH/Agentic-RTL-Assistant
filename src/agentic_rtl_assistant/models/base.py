"""Model provider protocol."""

from typing import Protocol

from agentic_rtl_assistant.models.types import ModelRequest, ModelResponse


class ModelProvider(Protocol):
    name: str

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a normalized model response."""
        ...
