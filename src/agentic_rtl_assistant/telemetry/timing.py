from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimingMetrics:
    total_seconds: float = 0.0
    model_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    validation_seconds: float = 0.0
