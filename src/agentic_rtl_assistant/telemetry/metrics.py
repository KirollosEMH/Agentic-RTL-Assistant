from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    tool_calls: int = 0
    retrieved_context_tokens: int = 0
    graph_nodes_visited: int = 0
    graph_edges_traversed: int = 0
