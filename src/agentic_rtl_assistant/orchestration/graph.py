from langgraph.graph import END, START, StateGraph

from agentic_rtl_assistant.orchestration.nodes import WorkflowNodes
from agentic_rtl_assistant.orchestration.routing import (
    route_after_classify,
    route_after_retrieval,
    route_after_validation,
)
from agentic_rtl_assistant.orchestration.state import AgentState


def build_workflow(nodes: WorkflowNodes, *, max_repair_attempts: int):
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("graph_retrieve", nodes.graph_retrieve)
    graph.add_node("source_retrieve", nodes.source_retrieve)
    graph.add_node("explain", nodes.explain)
    graph.add_node("generate", nodes.generate)
    graph.add_node("validate", nodes.validate)
    graph.add_node("repair", nodes.repair)
    graph.add_node("write", nodes.write_generated_code)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_classify)
    graph.add_edge("graph_retrieve", "source_retrieve")
    graph.add_conditional_edges("source_retrieve", route_after_retrieval)
    graph.add_edge("explain", "finalize")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: route_after_validation(
            state, max_repair_attempts=max_repair_attempts
        ),
    )
    graph.add_edge("repair", "validate")
    graph.add_edge("write", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()
