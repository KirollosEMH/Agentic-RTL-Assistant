"""GraphRAG: graph traversal followed by exact source-range retrieval."""

from __future__ import annotations

import time

from agentic_rtl_assistant.knowledge.evidence import (
    EvidencePack,
    GraphRelation,
    RetrievalMetrics,
    SourceEvidence,
)
from agentic_rtl_assistant.knowledge.graph.model import NodeType
from agentic_rtl_assistant.knowledge.graph.query import GraphQuery
from agentic_rtl_assistant.rtl.repository import RTLRepository


class GraphRetriever:
    def __init__(
        self,
        repository: RTLRepository,
        query: GraphQuery,
        *,
        max_hops: int = 2,
        max_nodes: int = 24,
        max_source_chunks: int = 6,
        max_context_tokens: int = 4000,
        include_graph_facts: bool = True,
        include_source_evidence: bool = True,
    ) -> None:
        self.repository = repository
        self.query = query
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.max_source_chunks = max_source_chunks
        self.max_context_tokens = max_context_tokens
        self.include_graph_facts = include_graph_facts
        self.include_source_evidence = include_source_evidence

    def retrieve(self, question: str) -> EvidencePack:
        started = time.perf_counter()
        seeds = self.query.resolve_entities(question)
        neighborhood = self.query.neighborhood(
            seeds, max_hops=self.max_hops, max_nodes=self.max_nodes
        )
        by_id = {node.id: node for node in neighborhood.nodes}
        relations = tuple(
            GraphRelation(
                by_id[edge.source].label,
                edge.relation.value,
                by_id[edge.target].label,
            )
            for edge in neighborhood.edges
            if self.include_graph_facts and edge.source in by_id and edge.target in by_id
        )
        evidence: list[SourceEvidence] = []
        seen_locations: set[tuple[str, int, int]] = set()
        source_tokens = 0
        if self.include_source_evidence:
            prioritized = sorted(
                neighborhood.nodes,
                key=lambda node: node.type is not NodeType.MODULE,
            )
            for node in prioritized:
                location = node.location
                if location is None:
                    continue
                key = (location.path, location.start_line, location.end_line)
                if key in seen_locations:
                    continue
                content = self.repository.read_lines(
                    location.path, location.start_line, location.end_line
                )
                words = len(content.split())
                if evidence and source_tokens + words > self.max_context_tokens:
                    break
                seen_locations.add(key)
                evidence.append(
                    SourceEvidence(
                        location.path,
                        location.start_line,
                        location.end_line,
                        content,
                        "graphrag",
                        node.label,
                    )
                )
                source_tokens += words
                if len(evidence) >= self.max_source_chunks:
                    break
        entities = tuple(
            node.label
            for node in neighborhood.nodes
            if node.type in {NodeType.MODULE, NodeType.INSTANCE, NodeType.PORT}
        )
        return EvidencePack(
            entities=entities,
            relations=relations,
            source_evidence=tuple(evidence),
            metrics=RetrievalMetrics(
                graph_nodes_visited=len(neighborhood.nodes),
                graph_edges_traversed=len(neighborhood.edges),
                graph_hops_used=neighborhood.hops_used,
                source_chunks_retrieved=len(evidence),
                source_tokens_retrieved=source_tokens,
                retrieval_latency_seconds=time.perf_counter() - started,
            ),
        )
