from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import fmean


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_]+", value.casefold()))


def _entity_recall(expected: tuple[str, ...], text: str) -> float | None:
    normalized_expected = {_normalized(item) for item in expected if _normalized(item)}
    if not normalized_expected:
        return None
    normalized_text = _normalized(text)
    return sum(item in normalized_text for item in normalized_expected) / len(normalized_expected)


def _fact_recall(expected: tuple[str, ...], text: str) -> float | None:
    facts = [set(_normalized(item).split()) for item in expected if _normalized(item)]
    if not facts:
        return None
    answer_terms = set(_normalized(text).split())
    return sum(fact <= answer_terms for fact in facts) / len(facts)


@dataclass(frozen=True, slots=True)
class CorrectnessScores:
    correctness: float | None
    answer_entity_recall: float | None
    answer_fact_recall: float | None
    rtl_structural_accuracy: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def score_correctness(
    answer: str | None,
    *,
    expected_answer_entities: list[str] | tuple[str, ...] = (),
    expected_answer_facts: list[str] | tuple[str, ...] = (),
    rtl_structural_accuracy: float | None = None,
) -> CorrectnessScores:
    """Score explicit answer labels or independently checked RTL structure.

    Facts are deterministic bags of required terms. Evaluation datasets should therefore use
    short, essential facts rather than complete reference-answer sentences.
    """

    text = answer or ""
    entity_recall = _entity_recall(tuple(expected_answer_entities), text)
    fact_recall = _fact_recall(tuple(expected_answer_facts), text)
    components = [
        score
        for score in (entity_recall, fact_recall, rtl_structural_accuracy)
        if score is not None
    ]
    return CorrectnessScores(
        correctness=fmean(components) if components else None,
        answer_entity_recall=entity_recall,
        answer_fact_recall=fact_recall,
        rtl_structural_accuracy=rtl_structural_accuracy,
    )


def aggregate_correctness_scores(
    scores: list[CorrectnessScores],
) -> dict[str, float | int | None]:
    names = tuple(CorrectnessScores.__dataclass_fields__)
    aggregated: dict[str, float | int | None] = {
        "evaluated_requests": sum(score.correctness is not None for score in scores)
    }
    for name in names:
        values = [value for score in scores if (value := getattr(score, name)) is not None]
        aggregated[name] = fmean(values) if values else None
    return aggregated
