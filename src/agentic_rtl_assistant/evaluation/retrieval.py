from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from statistics import fmean

from agentic_rtl_assistant.knowledge.evidence import EvidencePack


def _normalized_text(value: str) -> str:
    return value.strip().casefold()


def _normalized_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().casefold()


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Per-request retrieval scores; ``None`` means no ground truth was supplied."""

    retrieval_accuracy: float | None
    source_precision_at_k: float | None
    source_recall_at_k: float | None
    source_f1_at_k: float | None
    source_hit_rate: float | None
    source_mrr: float | None
    entity_recall: float | None
    relation_recall: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def score_retrieval(
    evidence: EvidencePack,
    *,
    expected_source_paths: list[str] | tuple[str, ...] = (),
    expected_entities: list[str] | tuple[str, ...] = (),
    expected_relations: list[tuple[str, str, str]] | tuple[tuple[str, str, str], ...] = (),
) -> RetrievalScores:
    """Score retrieved sources, entities, and graph relations against explicit labels.

    ``retrieval_accuracy`` is macro recall across every supplied ground-truth category. It
    answers "how much required evidence was retrieved?" without allowing a large category to
    hide a miss in a smaller one.
    """

    retrieved_sources = tuple(
        dict.fromkeys(_normalized_path(item.path) for item in evidence.source_evidence)
    )
    expected_sources = {_normalized_path(path) for path in expected_source_paths}
    relevant_sources = expected_sources & set(retrieved_sources)

    source_precision = None
    source_recall = None
    source_f1 = None
    source_hit_rate = None
    source_mrr = None
    if expected_sources:
        source_precision = _safe_divide(len(relevant_sources), len(retrieved_sources))
        source_recall = _safe_divide(len(relevant_sources), len(expected_sources))
        source_f1 = (
            2 * source_precision * source_recall / (source_precision + source_recall)
            if source_precision + source_recall
            else 0.0
        )
        source_hit_rate = float(bool(relevant_sources))
        source_mrr = next(
            (
                1.0 / rank
                for rank, path in enumerate(retrieved_sources, start=1)
                if path in expected_sources
            ),
            0.0,
        )

    evidence_text = "\n".join(
        (
            *evidence.entities,
            *(item.content for item in evidence.source_evidence),
            *(
                f"{relation.source} {relation.relation} {relation.target}"
                for relation in evidence.relations
            ),
        )
    ).casefold()
    normalized_entities = {_normalized_text(entity) for entity in expected_entities}
    entity_recall = (
        _safe_divide(
            sum(entity in evidence_text for entity in normalized_entities),
            len(normalized_entities),
        )
        if normalized_entities
        else None
    )

    retrieved_relations = {
        tuple(_normalized_text(part) for part in (item.source, item.relation, item.target))
        for item in evidence.relations
    }
    normalized_relations = {
        tuple(_normalized_text(part) for part in relation) for relation in expected_relations
    }
    relation_recall = (
        _safe_divide(len(normalized_relations & retrieved_relations), len(normalized_relations))
        if normalized_relations
        else None
    )

    recall_components = [
        score for score in (source_recall, entity_recall, relation_recall) if score is not None
    ]
    retrieval_accuracy = fmean(recall_components) if recall_components else None
    return RetrievalScores(
        retrieval_accuracy=retrieval_accuracy,
        source_precision_at_k=source_precision,
        source_recall_at_k=source_recall,
        source_f1_at_k=source_f1,
        source_hit_rate=source_hit_rate,
        source_mrr=source_mrr,
        entity_recall=entity_recall,
        relation_recall=relation_recall,
    )


def aggregate_retrieval_scores(scores: list[RetrievalScores]) -> dict[str, float | int | None]:
    """Macro-average each available retrieval metric across evaluated requests."""

    names = tuple(RetrievalScores.__dataclass_fields__)
    aggregated: dict[str, float | int | None] = {
        "evaluated_requests": sum(score.retrieval_accuracy is not None for score in scores)
    }
    for name in names:
        values = [value for score in scores if (value := getattr(score, name)) is not None]
        aggregated[name] = fmean(values) if values else None
    return aggregated
