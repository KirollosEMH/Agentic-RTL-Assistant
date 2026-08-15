from agentic_rtl_assistant.evaluation.correctness import (
    CorrectnessScores,
    aggregate_correctness_scores,
    score_correctness,
)


def contains_expected_entities(answer: str, expected_entities: list[str]) -> bool:
    lowered = answer.casefold()
    return all(entity.casefold() in lowered for entity in expected_entities)


__all__ = [
    "CorrectnessScores",
    "aggregate_correctness_scores",
    "contains_expected_entities",
    "score_correctness",
]
