"""A2 baseline boundary. Tool-loop expansion is intentionally deferred."""

from agentic_rtl_assistant.approaches.direct import DirectLLMApproach


class SingleAgentApproach(DirectLLMApproach):
    name = "single_agent"
