from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.approaches.base import RunResult
from agentic_rtl_assistant.config.models import AppConfig, ApproachType, ModelProfile
from agentic_rtl_assistant.evaluation.correctness import (
    CorrectnessScores,
    aggregate_correctness_scores,
    score_correctness,
)
from agentic_rtl_assistant.evaluation.grounding import (
    GroundingScores,
    aggregate_grounding_scores,
    score_grounding,
)
from agentic_rtl_assistant.evaluation.retrieval import (
    RetrievalScores,
    aggregate_retrieval_scores,
    score_retrieval,
)
from agentic_rtl_assistant.evaluation.validation import (
    ExpectedInstance,
    ExpectedPort,
    ValidationScores,
    aggregate_validation_scores,
    extract_verilog,
    score_validation,
)
from agentic_rtl_assistant.knowledge.evidence import EvidencePack
from agentic_rtl_assistant.rtl.parser import PyVerilogParser


class EvaluationCase(BaseModel):
    id: str
    type: Literal["qa", "code_generation"]
    prompt: str
    expected_source_paths: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)
    expected_relations: list[tuple[str, str, str]] = Field(default_factory=list)
    expected_answer_entities: list[str] = Field(default_factory=list)
    expected_answer_facts: list[str] = Field(default_factory=list)
    expected_module: str | None = None
    expected_ports: list[ExpectedPort] = Field(default_factory=list)
    expected_instances: list[ExpectedInstance] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    cases: list[EvaluationCase]


@dataclass(frozen=True, slots=True)
class EvaluationCell:
    approach: ApproachType
    profile_name: str
    profile: ModelProfile
    config: AppConfig

    @property
    def identifier(self) -> str:
        raw = f"{self.approach.value}--{self.profile_name}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


@dataclass(frozen=True, slots=True)
class EvaluationScores:
    correctness: CorrectnessScores | None = None
    grounding: GroundingScores | None = None
    retrieval: RetrievalScores | None = None
    validation: ValidationScores | None = None


def load_dataset(path: Path) -> EvaluationDataset:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(data)


def _config_hash(config: AppConfig) -> tuple[str, str]:
    serialized = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, digest


def _unique[T](values: list[T]) -> list[T]:
    return list(dict.fromkeys(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def task_succeeded(
    case: EvaluationCase,
    *,
    execution_success: bool,
    correctness: CorrectnessScores | None,
    validation: ValidationScores | None,
) -> bool:
    """Apply deterministic dataset contracts independently of process success."""

    if not execution_success:
        return False
    if case.type == "code_generation":
        return validation is not None and validation.validation_accuracy == 1.0
    if correctness is not None and correctness.correctness is not None:
        return correctness.correctness == 1.0
    return True


class EvaluationRunner:
    def __init__(
        self,
        config: AppConfig,
        service_factory: Callable[[AppConfig], ApplicationService] = ApplicationService,
    ) -> None:
        self.config = config
        self.service_factory = service_factory

    def _model_profiles(self) -> list[str]:
        configured = _unique(self.config.evaluation.model_profiles)
        if configured:
            return configured
        explanation = self.config.orchestration.agents.get("rtl_explanation")
        if explanation is None:
            raise ValueError("evaluation needs a model profile or rtl_explanation agent")
        return [explanation.model]

    def _approaches(self) -> list[ApproachType]:
        configured = _unique(self.config.evaluation.approaches)
        return [ApproachType(value) for value in configured] or [self.config.approach.type]

    def _cell(self, approach: ApproachType, profile_name: str) -> EvaluationCell:
        profile = self.config.models.profiles[profile_name]
        agents = {
            name: settings.model_copy(update={"model": profile_name})
            for name, settings in self.config.orchestration.agents.items()
        }
        candidate = self.config.model_copy(
            update={
                "approach": self.config.approach.model_copy(update={"type": approach}),
                "orchestration": self.config.orchestration.model_copy(
                    update={"agents": agents}
                ),
                "evaluation": self.config.evaluation.model_copy(
                    update={"approaches": [approach], "model_profiles": [profile_name]}
                ),
            }
        )
        cell_config = AppConfig.model_validate(candidate.model_dump())
        return EvaluationCell(approach, profile_name, profile, cell_config)

    def _cells(self) -> list[EvaluationCell]:
        return [
            self._cell(approach, profile_name)
            for approach in self._approaches()
            for profile_name in self._model_profiles()
        ]

    def _result_record(
        self,
        cell: EvaluationCell,
        case: EvaluationCase,
        repetition: int,
        result: RunResult | None,
        error: str | None = None,
    ) -> tuple[dict[str, object], EvaluationScores]:
        execution_success = result is not None and result.succeeded and error is None
        record: dict[str, object] = {
            "approach": cell.approach.value,
            "model_profile": cell.profile_name,
            "provider": cell.profile.provider,
            "model": cell.profile.model,
            "case_id": case.id,
            "repetition": repetition,
            "success": execution_success,
            "execution_success": execution_success,
            "answer": result.answer if result is not None else None,
            "generated_code": result.generated_code if result is not None else None,
            "written_files": list(result.written_files) if result is not None else [],
            "write_error": result.write_error if result is not None else None,
            "usage": {
                "input_tokens": result.usage.input_tokens if result is not None else 0,
                "cached_input_tokens": (
                    result.usage.cached_input_tokens if result is not None else None
                ),
                "output_tokens": result.usage.output_tokens if result is not None else 0,
                "llm_calls": result.usage.llm_calls if result is not None else 0,
            },
            "context": {
                "latest_input_tokens": (
                    result.context_window.latest_input_tokens if result is not None else 0
                ),
                "peak_input_tokens": (
                    result.context_window.peak_input_tokens if result is not None else 0
                ),
                "history_messages": (
                    result.context_window.history_messages if result is not None else 0
                ),
            },
            "latency_seconds": result.timing.total_seconds if result is not None else 0.0,
            "error": error or (result.error if result is not None else None),
        }
        evidence = result.evidence if result is not None else EvidencePack()
        answer = result.answer if result is not None else None

        retrieval = None
        if "retrieval" in self.config.evaluation.metrics:
            retrieval = score_retrieval(
                evidence,
                expected_source_paths=case.expected_source_paths,
                expected_entities=case.expected_entities,
                expected_relations=case.expected_relations,
            )
            record["retrieval"] = {
                "retrieved_source_paths": list(
                    dict.fromkeys(item.path for item in evidence.source_evidence)
                ),
                "retrieved_entities": list(evidence.entities),
                "retrieved_relations": [
                    [item.source, item.relation, item.target]
                    for item in evidence.relations
                ],
                "scores": retrieval.as_dict(),
            }

        validation = None
        if case.type == "code_generation":
            generated_code = result.generated_code if result is not None else None
            source = generated_code or extract_verilog(answer)
            validation = score_validation(
                source,
                parser=PyVerilogParser(),
                runtime_validation=result.validation if result is not None else None,
                expected_module=case.expected_module,
                expected_ports=case.expected_ports,
                expected_instances=case.expected_instances,
            )
            if "validation" in self.config.evaluation.metrics:
                record["validation"] = {
                    "source": (
                        "generated_code"
                        if generated_code
                        else "answer" if source is not None else None
                    ),
                    "scores": validation.as_dict(),
                }

        correctness = score_correctness(
            answer,
            expected_answer_entities=case.expected_answer_entities,
            expected_answer_facts=case.expected_answer_facts,
            rtl_structural_accuracy=(
                validation.validation_accuracy if validation is not None else None
            ),
        )
        if "correctness" in self.config.evaluation.metrics:
            record["correctness"] = {"scores": correctness.as_dict()}

        grounding = None
        if "grounding" in self.config.evaluation.metrics and case.type == "qa":
            grounding = score_grounding(
                answer,
                evidence,
                project_root=cell.config.project.root,
                expected_source_paths=case.expected_source_paths,
                expected_answer_entities=case.expected_answer_entities,
            )
            record["grounding"] = {
                "scores": grounding.as_dict(),
            }

        record["task_success"] = task_succeeded(
            case,
            execution_success=execution_success,
            correctness=correctness,
            validation=validation,
        )

        return record, EvaluationScores(correctness, grounding, retrieval, validation)

    def _metrics(
        self,
        results: list[dict[str, object]],
        scores: list[EvaluationScores],
    ) -> dict[str, object]:
        input_tokens = 0
        cached_tokens = 0
        cached_tokens_complete = True
        output_tokens = 0
        llm_calls = 0
        latest_context_tokens: list[int] = []
        peak_context_tokens: list[int] = []
        history_messages: list[int] = []
        latencies: list[float] = []
        for result in results:
            usage = result["usage"]
            assert isinstance(usage, dict)
            input_tokens += int(usage["input_tokens"])
            output_tokens += int(usage["output_tokens"])
            llm_calls += int(usage["llm_calls"])
            cached = usage["cached_input_tokens"]
            if cached is None:
                cached_tokens_complete = False
            else:
                cached_tokens += int(cached)
            context = result["context"]
            assert isinstance(context, dict)
            latest_context_tokens.append(int(context["latest_input_tokens"]))
            peak_context_tokens.append(int(context["peak_input_tokens"]))
            history_messages.append(int(context["history_messages"]))
            latencies.append(float(result["latency_seconds"]))

        requests = len(results)
        execution_successes = sum(
            bool(result.get("execution_success", result["success"])) for result in results
        )
        task_successes = sum(bool(result["task_success"]) for result in results)
        total_tokens = input_tokens + output_tokens
        latency_seconds = sum(latencies)

        metrics: dict[str, object] = {
            "requests": requests,
            "successes": execution_successes,
            "success_rate": execution_successes / requests if requests else 0.0,
            "execution_successes": execution_successes,
            "execution_success_rate": (
                execution_successes / requests if requests else 0.0
            ),
            "task_successes": task_successes,
            "task_success_rate": task_successes / requests if requests else 0.0,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens if cached_tokens_complete else None,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "llm_calls": llm_calls,
            "latency_seconds": latency_seconds,
            "average_input_tokens_per_request": input_tokens / requests if requests else 0.0,
            "average_output_tokens_per_request": output_tokens / requests if requests else 0.0,
            "average_total_tokens_per_request": total_tokens / requests if requests else 0.0,
            "average_tokens_per_llm_call": total_tokens / llm_calls if llm_calls else 0.0,
            "cached_input_token_ratio": (
                cached_tokens / input_tokens
                if cached_tokens_complete and input_tokens
                else None
            ),
            "average_latest_context_tokens": (
                fmean(latest_context_tokens) if latest_context_tokens else 0.0
            ),
            "average_peak_context_tokens": (
                fmean(peak_context_tokens) if peak_context_tokens else 0.0
            ),
            "max_peak_context_tokens": max(peak_context_tokens, default=0),
            "average_history_messages": (
                fmean(history_messages) if history_messages else 0.0
            ),
            "average_latency_seconds": fmean(latencies) if latencies else 0.0,
            "p50_latency_seconds": _percentile(latencies, 0.50),
            "p95_latency_seconds": _percentile(latencies, 0.95),
        }
        if "correctness" in self.config.evaluation.metrics:
            metrics["correctness"] = aggregate_correctness_scores(
                [score.correctness for score in scores if score.correctness is not None]
            )
        if "grounding" in self.config.evaluation.metrics:
            metrics["grounding"] = aggregate_grounding_scores(
                [score.grounding for score in scores if score.grounding is not None]
            )
        if "retrieval" in self.config.evaluation.metrics:
            metrics["retrieval"] = aggregate_retrieval_scores(
                [score.retrieval for score in scores if score.retrieval is not None]
            )
        if "validation" in self.config.evaluation.metrics:
            metrics["validation"] = aggregate_validation_scores(
                [score.validation for score in scores if score.validation is not None]
            )
        return metrics

    async def _run_cell(
        self, cell: EvaluationCell
    ) -> tuple[list[dict[str, object]], list[EvaluationScores], list[dict[str, object]]]:
        results: list[dict[str, object]] = []
        scores: list[EvaluationScores] = []
        traces: list[dict[str, object]] = []
        service: ApplicationService | None = None
        startup_error: str | None = None
        try:
            service = self.service_factory(cell.config)
        except Exception as exc:  # one invalid matrix cell must not abort the remaining cells
            startup_error = f"{type(exc).__name__}: {exc}"

        for dataset_path in cell.config.evaluation.datasets:
            dataset = load_dataset(dataset_path)
            for case in dataset.cases:
                for repetition in range(cell.config.evaluation.repetitions):
                    result = None
                    error = startup_error
                    if service is not None:
                        try:
                            result = await service.ask(case.prompt)
                        except Exception as exc:  # preserve partial matrix results
                            error = f"{type(exc).__name__}: {exc}"
                    record, result_scores = self._result_record(
                        cell, case, repetition, result, error
                    )
                    results.append(record)
                    scores.append(result_scores)

        if service is not None:
            for event in service.telemetry.events:
                trace = event.as_dict()
                trace.update(
                    {
                        "approach": cell.approach.value,
                        "model_profile": cell.profile_name,
                        "provider": cell.profile.provider,
                        "model": cell.profile.model,
                    }
                )
                traces.append(trace)
        return results, scores, traces

    async def run(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_directory = self.config.telemetry.persistence_path / f"{stamp}-evaluation-matrix"
        run_directory.mkdir(parents=True, exist_ok=False)
        serialized, config_hash = _config_hash(self.config)
        (run_directory / "config.yaml").write_text(serialized, encoding="utf-8")

        all_results: list[dict[str, object]] = []
        all_scores: list[EvaluationScores] = []
        all_traces: list[dict[str, object]] = []
        comparisons: list[dict[str, object]] = []
        combinations_directory = run_directory / "combinations"
        combinations_directory.mkdir()

        cells = self._cells()
        for cell in cells:
            results, scores, traces = await self._run_cell(cell)
            cell_metrics = self._metrics(results, scores)
            cell_metrics.update(
                {
                    "approach": cell.approach.value,
                    "model_profile": cell.profile_name,
                    "provider": cell.profile.provider,
                    "model": cell.profile.model,
                }
            )
            comparisons.append(cell_metrics)
            all_results.extend(results)
            all_scores.extend(scores)
            all_traces.extend(traces)

            cell_directory = combinations_directory / cell.identifier
            cell_directory.mkdir()
            cell_config, _ = _config_hash(cell.config)
            (cell_directory / "config.yaml").write_text(cell_config, encoding="utf-8")
            (cell_directory / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            (cell_directory / "metrics.json").write_text(
                json.dumps(cell_metrics, indent=2), encoding="utf-8"
            )
            cell_trace_text = "\n".join(json.dumps(trace) for trace in traces)
            (cell_directory / "traces.jsonl").write_text(
                cell_trace_text + ("\n" if cell_trace_text else ""), encoding="utf-8"
            )

        metrics = self._metrics(all_results, all_scores)
        metrics.update(
            {
                "config_hash": config_hash,
                "matrix": {
                    "approaches": [approach.value for approach in self._approaches()],
                    "model_profiles": self._model_profiles(),
                    "combinations": len(cells),
                },
                "comparisons": comparisons,
            }
        )
        (run_directory / "results.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        (run_directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        trace_text = "\n".join(json.dumps(trace) for trace in all_traces)
        (run_directory / "traces.jsonl").write_text(
            trace_text + ("\n" if trace_text else ""), encoding="utf-8"
        )
        return run_directory
