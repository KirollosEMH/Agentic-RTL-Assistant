from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agentic_rtl_assistant.models.types import TokenUsage


@dataclass(frozen=True, slots=True)
class ContextWindowMetrics:
    latest_input_tokens: int = 0
    peak_input_tokens: int = 0
    history_messages: int = 0

    @classmethod
    def from_usage_events(
        cls,
        items: Iterable[TokenUsage],
        *,
        history_messages: int = 0,
    ) -> ContextWindowMetrics:
        values = list(items)
        return cls(
            latest_input_tokens=values[-1].input_tokens if values else 0,
            peak_input_tokens=max((item.input_tokens for item in values), default=0),
            history_messages=history_messages,
        )
