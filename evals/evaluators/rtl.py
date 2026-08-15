from agentic_rtl_assistant.evaluation.validation import (
    ExpectedInstance,
    ExpectedPort,
    ValidationScores,
    aggregate_validation_scores,
    extract_verilog,
    score_validation,
)
from agentic_rtl_assistant.rtl.types import ValidationResult


def validation_score(result: ValidationResult | None) -> float:
    return 1.0 if result is not None and result.valid else 0.0


__all__ = [
    "ExpectedInstance",
    "ExpectedPort",
    "ValidationScores",
    "aggregate_validation_scores",
    "extract_verilog",
    "score_validation",
    "validation_score",
]
