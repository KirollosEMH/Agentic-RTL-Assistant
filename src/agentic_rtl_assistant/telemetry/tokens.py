from collections.abc import Iterable

from agentic_rtl_assistant.models.types import TokenUsage


def aggregate_usage(items: Iterable[TokenUsage]) -> TokenUsage:
    values = list(items)
    if not values:
        return TokenUsage()
    cached = (
        sum(item.cached_input_tokens or 0 for item in values)
        if all(item.cached_input_tokens is not None for item in values)
        else None
    )
    return TokenUsage(
        input_tokens=sum(item.input_tokens for item in values),
        output_tokens=sum(item.output_tokens for item in values),
        cached_input_tokens=cached,
        llm_calls=sum(item.llm_calls for item in values),
    )
