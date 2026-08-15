from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    path: str
    start_line: int | None
    end_line: int | None
    content: str
    retrieval_type: str
    related_entity: str | None = None


@dataclass(frozen=True, slots=True)
class GraphRelation:
    source: str
    relation: str
    target: str


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    graph_nodes_visited: int = 0
    graph_edges_traversed: int = 0
    graph_hops_used: int = 0
    source_chunks_retrieved: int = 0
    source_tokens_retrieved: int = 0
    retrieval_latency_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class EvidencePack:
    entities: tuple[str, ...] = ()
    relations: tuple[GraphRelation, ...] = ()
    source_evidence: tuple[SourceEvidence, ...] = ()
    metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)

    @property
    def estimated_tokens(self) -> int:
        text = " ".join(source.content for source in self.source_evidence)
        text += " " + " ".join(
            f"{relation.source} {relation.relation} {relation.target}"
            for relation in self.relations
        )
        return max(0, len(text.split()))

    def to_prompt(self) -> str:
        relation_lines = [
            f"- {relation.source} {relation.relation} {relation.target}"
            for relation in self.relations
        ]
        source_blocks = [
            f"[{item.path}:{item.start_line or '?'}-{item.end_line or '?'}]\n{item.content}"
            for item in self.source_evidence
        ]
        return (
            "Entities:\n- "
            + "\n- ".join(self.entities)
            + "\n\nGraph facts:\n"
            + ("\n".join(relation_lines) or "(none)")
            + "\n\nSource evidence:\n"
            + ("\n\n".join(source_blocks) or "(none)")
        )
