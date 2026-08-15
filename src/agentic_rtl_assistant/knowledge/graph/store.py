from __future__ import annotations

from typing import Protocol

from agentic_rtl_assistant.knowledge.graph.model import GraphEdge, GraphNode, NodeType


class ProjectKnowledgeGraph(Protocol):
    def clear(self) -> None: ...

    def add_node(self, node: GraphNode) -> None: ...

    def add_edge(self, edge: GraphEdge) -> None: ...

    def get_node(self, node_id: str) -> GraphNode | None: ...

    def nodes(self, node_type: NodeType | None = None) -> tuple[GraphNode, ...]: ...

    def edges(self) -> tuple[GraphEdge, ...]: ...

    def adjacent_edges(self, node_id: str) -> tuple[GraphEdge, ...]: ...


class InMemoryGraphStore:
    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if edge not in self._edges:
            self._edges.append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def nodes(self, node_type: NodeType | None = None) -> tuple[GraphNode, ...]:
        values = self._nodes.values()
        return tuple(node for node in values if node_type is None or node.type is node_type)

    def edges(self) -> tuple[GraphEdge, ...]:
        return tuple(self._edges)

    def adjacent_edges(self, node_id: str) -> tuple[GraphEdge, ...]:
        return tuple(
            edge for edge in self._edges if edge.source == node_id or edge.target == node_id
        )
