from agentic_rtl_assistant.models.types import TokenUsage


def total_tokens(usage: TokenUsage) -> int:
    return usage.total_tokens
