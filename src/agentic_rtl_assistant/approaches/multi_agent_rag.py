"""A3 uses the same bounded multi-agent workflow with text evidence only."""

from agentic_rtl_assistant.approaches.multi_agent_graphrag import MultiAgentGraphRAGApproach


class MultiAgentRAGApproach(MultiAgentGraphRAGApproach):
    name = "multi_agent_rag"
