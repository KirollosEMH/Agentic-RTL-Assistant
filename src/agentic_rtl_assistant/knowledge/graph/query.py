from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from agentic_rtl_assistant.knowledge.graph.model import GraphEdge, GraphNode, NodeType
from agentic_rtl_assistant.knowledge.graph.store import ProjectKnowledgeGraph


@dataclass(frozen=True, slots=True)
class Neighborhood:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    hops_used: int


class GraphQuery:
    def __init__(self, store: ProjectKnowledgeGraph) -> None:
        self.store = store

    def resolve_entities(self, question: str) -> tuple[GraphNode, ...]:
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question.casefold()))
        matches = [
            node
            for node in self.store.nodes()
            if node.label.casefold() in words or node.label.casefold() in question.casefold()
        ]
        if not matches and {"module", "modules"} & words:
            matches = list(self.store.nodes(NodeType.MODULE))
        return tuple(matches)

    def neighborhood(
        self,
        seeds: tuple[GraphNode, ...],
        *,
        max_hops: int,
        max_nodes: int,
    ) -> Neighborhood:
        visited: dict[str, GraphNode] = {node.id: node for node in seeds[:max_nodes]}
        queue = deque((node.id, 0) for node in seeds[:max_nodes])
        selected_edges: list[GraphEdge] = []
        hops_used = 0
        while queue and len(visited) < max_nodes:
            node_id, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for edge in self.store.adjacent_edges(node_id):
                if edge not in selected_edges:
                    selected_edges.append(edge)
                neighbor_id = edge.target if edge.source == node_id else edge.source
                if neighbor_id in visited:
                    continue
                neighbor = self.store.get_node(neighbor_id)
                if neighbor is None:
                    continue
                visited[neighbor_id] = neighbor
                hops_used = max(hops_used, depth + 1)
                if len(visited) >= max_nodes:
                    break
                queue.append((neighbor_id, depth + 1))
        return Neighborhood(tuple(visited.values()), tuple(selected_edges), hops_used)
