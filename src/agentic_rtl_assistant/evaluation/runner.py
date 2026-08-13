"""Evaluation runner kept outside the production request graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.config.models import AppConfig
from agentic_rtl_assistant.evaluation.retrieval import (
    RetrievalScores,
    aggregate_retrieval_scores,
    score_retrieval,
)


class EvaluationCase(BaseModel):
    id: str
    type: Literal["qa", "code_generation"]
    prompt: str
    expected_source_paths: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)
    expected_relations: list[tuple[str, str, str]] = Field(default_factory=list)
    expected_module: str | None = None


class EvaluationDataset(BaseModel):
    cases: list[EvaluationCase]


def load_dataset(path: Path) -> EvaluationDataset:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(data)


class EvaluationRunner:
    def __init__(
        self,
        config: AppConfig,
        service_factory: Callable[[AppConfig], ApplicationService] = ApplicationService,
    ) -> None:
        self.config = config
        self.service_factory = service_factory

    async def run(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{self.config.approach.type.value}"
        run_directory = self.config.telemetry.persistence_path / run_id
        run_directory.mkdir(parents=True, exist_ok=False)
        resolved = self.config.model_dump(mode="json")
        serialized = yaml.safe_dump(resolved, sort_keys=True)
        config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        (run_directory / "config.yaml").write_text(serialized, encoding="utf-8")

        service = self.service_factory(self.config)
        results: list[dict[str, object]] = []
        retrieval_scores: list[RetrievalScores] = []
        for dataset_path in self.config.evaluation.datasets:
            dataset = load_dataset(dataset_path)
            for case in dataset.cases:
                for repetition in range(self.config.evaluation.repetitions):
                    result = await service.ask(case.prompt)
                    result_record: dict[str, object] = {
                        "case_id": case.id,
                        "repetition": repetition,
                        "success": result.succeeded,
                        "answer": result.answer,
                        "generated_code": result.generated_code,
                        "usage": {
                            "input_tokens": result.usage.input_tokens,
                            "cached_input_tokens": result.usage.cached_input_tokens,
                            "output_tokens": result.usage.output_tokens,
                            "llm_calls": result.usage.llm_calls,
                        },
                        "latency_seconds": result.timing.total_seconds,
                        "error": result.error,
                    }
                    if "retrieval" in self.config.evaluation.metrics:
                        scores = score_retrieval(
                            result.evidence,
                            expected_source_paths=case.expected_source_paths,
                            expected_entities=case.expected_entities,
                            expected_relations=case.expected_relations,
                        )
                        retrieval_scores.append(scores)
                        result_record["retrieval"] = {
                            "retrieved_source_paths": list(
                                dict.fromkeys(item.path for item in result.evidence.source_evidence)
                            ),
                            "retrieved_entities": list(result.evidence.entities),
                            "retrieved_relations": [
                                [item.source, item.relation, item.target]
                                for item in result.evidence.relations
                            ],
                            "scores": scores.as_dict(),
                        }
                    results.append(result_record)
        metrics = {
            "config_hash": config_hash,
            "requests": len(results),
            "successes": sum(bool(result["success"]) for result in results),
            "input_tokens": sum(int(result["usage"]["input_tokens"]) for result in results),
            "output_tokens": sum(int(result["usage"]["output_tokens"]) for result in results),
        }
        if "retrieval" in self.config.evaluation.metrics:
            metrics["retrieval"] = aggregate_retrieval_scores(retrieval_scores)
        (run_directory / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (run_directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        traces = "\n".join(json.dumps(event.as_dict()) for event in service.telemetry.events)
        (run_directory / "traces.jsonl").write_text(
            traces + ("\n" if traces else ""), encoding="utf-8"
        )
        return run_directory
