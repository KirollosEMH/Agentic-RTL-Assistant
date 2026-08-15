from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int | None = None
    llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        cached = (
            self.cached_input_tokens + other.cached_input_tokens
            if self.cached_input_tokens is not None and other.cached_input_tokens is not None
            else None
        )
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=cached,
            llm_calls=self.llm_calls + other.llm_calls,
        )


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model: str
    messages: tuple[ModelMessage, ...]
    temperature: float = 0.0
    max_output_tokens: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    provider: str
    model: str
    usage: TokenUsage
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProviderError(RuntimeError):
    """Normalized provider failure."""
