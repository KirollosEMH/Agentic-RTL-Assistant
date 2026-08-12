"""Compatibility import for running the evaluator directly from the repository."""

from agentic_rtl_assistant.evaluation.runner import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationRunner,
    load_dataset,
)

__all__ = ["EvaluationCase", "EvaluationDataset", "EvaluationRunner", "load_dataset"]
