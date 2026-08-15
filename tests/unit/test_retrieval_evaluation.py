import json
from pathlib import Path

import pytest

from agentic_rtl_assistant.approaches.base import RunResult
from agentic_rtl_assistant.config import ApproachType, EvaluationMetric, load_config
from agentic_rtl_assistant.evaluation.retrieval import (
    aggregate_retrieval_scores,
    score_retrieval,
)
from agentic_rtl_assistant.evaluation.runner import EvaluationRunner, load_dataset
from agentic_rtl_assistant.knowledge.evidence import (
    EvidencePack,
    GraphRelation,
    SourceEvidence,
)
from agentic_rtl_assistant.models.types import TokenUsage
from agentic_rtl_assistant.telemetry.context import ContextWindowMetrics
from agentic_rtl_assistant.telemetry.timing import TimingMetrics


def _source(path: str, content: str = "") -> SourceEvidence:
    return SourceEvidence(path, 1, 2, content, "test")


def test_retrieval_scores_sources_entities_relations_and_rank() -> None:
    evidence = EvidencePack(
        entities=("DataPipeline",),
        relations=(GraphRelation("DataPipeline", "INSTANTIATES", "CounterProducer"),),
        source_evidence=(
            _source("rtl/data_pipeline.v", "input wire ready_in;"),
            _source("rtl/unrelated.v"),
            _source("rtl/helper.v"),
        ),
    )

    scores = score_retrieval(
        evidence,
        expected_source_paths=["rtl\\data_pipeline.v", "rtl/missing.v"],
        expected_entities=["DataPipeline", "ready_in", "missing_signal"],
        expected_relations=[
            ("DataPipeline", "INSTANTIATES", "CounterProducer"),
            ("DataPipeline", "INSTANTIATES", "DataConsumer"),
        ],
    )

    assert scores.source_precision_at_k == pytest.approx(1 / 3)
    assert scores.source_recall_at_k == 0.5
    assert scores.source_f1_at_k == pytest.approx(0.4)
    assert scores.source_hit_rate == 1.0
    assert scores.source_mrr == 1.0
    assert scores.entity_recall == pytest.approx(2 / 3)
    assert scores.relation_recall == 0.5
    assert scores.retrieval_accuracy == pytest.approx((0.5 + 2 / 3 + 0.5) / 3)


def test_retrieval_aggregate_ignores_unlabelled_categories() -> None:
    labelled = score_retrieval(
        EvidencePack(source_evidence=(_source("expected.v"),)),
        expected_source_paths=["expected.v"],
    )
    unlabelled = score_retrieval(EvidencePack())

    aggregate = aggregate_retrieval_scores([labelled, unlabelled])

    assert aggregate["evaluated_requests"] == 1
    assert aggregate["retrieval_accuracy"] == 1.0
    assert aggregate["entity_recall"] is None


def test_extended_dataset_includes_integration_contract(repository_root: Path) -> None:
    dataset = load_dataset(repository_root / "evals" / "cases" / "extended.yaml")

    buffered = next(case for case in dataset.cases if case.id == "generate_buffered_pipeline")
    assert buffered.expected_module == "BufferedDataPipeline"
    assert {instance.module for instance in buffered.expected_instances} == {
        "CounterProducer",
        "FifoBuffer",
        "DataConsumer",
    }


class StubTelemetry:
    events = ()


class StubService:
    telemetry = StubTelemetry()

    async def ask(self, prompt: str) -> RunResult:
        return RunResult(
            request_id="request",
            approach="stub",
            answer=prompt,
            evidence=EvidencePack(
                entities=("DataPipeline",),
                source_evidence=(_source("data_pipeline.v", "module DataPipeline;"),),
            ),
        )


class QualityStubService:
    telemetry = StubTelemetry()

    async def ask(self, prompt: str) -> RunResult:
        if "Generate Demo" in prompt:
            return RunResult(
                request_id="code-request",
                approach="stub",
                answer="```verilog\nmodule Demo(input wire clk); endmodule\n```",
                usage=TokenUsage(20, 10, 5, 1),
                context_window=ContextWindowMetrics(20, 20, 2),
                timing=TimingMetrics(total_seconds=2.0),
            )
        return RunResult(
            request_id="qa-request",
            approach="stub",
            answer="DataPipeline is defined at [data_pipeline.v:1-2].",
            evidence=EvidencePack(
                entities=("DataPipeline",),
                source_evidence=(_source("data_pipeline.v", "module DataPipeline;"),),
            ),
            usage=TokenUsage(10, 5, 2, 1),
            context_window=ContextWindowMetrics(10, 10, 0),
            timing=TimingMetrics(total_seconds=1.0),
        )


@pytest.mark.asyncio
async def test_evaluation_runner_persists_retrieval_scores(
    repository_root: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "retrieval.yaml"
    dataset.write_text(
        """cases:
  - id: pipeline
    type: qa
    prompt: Find the pipeline
    expected_source_paths: [data_pipeline.v]
    expected_entities: [DataPipeline]
""",
        encoding="utf-8",
    )
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "datasets": [dataset],
                    "metrics": [EvaluationMetric.RETRIEVAL],
                    "approaches": [ApproachType.MULTI_AGENT_GRAPHRAG],
                    "model_profiles": ["reasoning"],
                }
            ),
            "telemetry": config.telemetry.model_copy(
                update={"persistence_path": tmp_path / "runs"}
            ),
        }
    )

    run_directory = await EvaluationRunner(config, lambda _config: StubService()).run()
    results = json.loads((run_directory / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))

    assert results[0]["retrieval"]["scores"]["retrieval_accuracy"] == 1.0
    assert results[0]["retrieval"]["retrieved_source_paths"] == ["data_pipeline.v"]
    assert metrics["retrieval"]["evaluated_requests"] == 1
    assert metrics["retrieval"]["source_precision_at_k"] == 1.0
    assert metrics["total_tokens"] == 0
    assert metrics["average_latency_seconds"] == 0.0
    assert metrics["p95_latency_seconds"] == 0.0


@pytest.mark.asyncio
async def test_evaluation_runner_builds_approach_model_cross_product(
    repository_root: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "matrix.yaml"
    dataset.write_text(
        """cases:
  - id: matrix_case
    type: qa
    prompt: Compare this model
""",
        encoding="utf-8",
    )
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    profiles = dict(config.models.profiles)
    profiles["openrouter_test"] = profiles["reasoning"].model_copy(
        update={"provider": "openrouter", "model": "test/openrouter-model"}
    )
    config = config.model_copy(
        update={
            "models": config.models.model_copy(update={"profiles": profiles}),
            "evaluation": config.evaluation.model_copy(
                update={
                    "datasets": [dataset],
                    "metrics": [],
                    "approaches": [ApproachType.DIRECT_LLM, ApproachType.TEXT_RAG],
                    "model_profiles": ["reasoning", "openrouter_test"],
                }
            ),
            "telemetry": config.telemetry.model_copy(
                update={"persistence_path": tmp_path / "runs"}
            ),
        }
    )
    selected_configs = []

    def service_factory(selected_config):
        selected_configs.append(selected_config)
        return StubService()

    run_directory = await EvaluationRunner(config, service_factory).run()
    results = json.loads((run_directory / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))

    assert {
        (result["approach"], result["model_profile"], result["provider"], result["model"])
        for result in results
    } == {
        ("direct_llm", "reasoning", "ollama", "gpt-oss:120b"),
        ("direct_llm", "openrouter_test", "openrouter", "test/openrouter-model"),
        ("text_rag", "reasoning", "ollama", "gpt-oss:120b"),
        ("text_rag", "openrouter_test", "openrouter", "test/openrouter-model"),
    }
    assert metrics["matrix"]["combinations"] == 4
    assert len(metrics["comparisons"]) == 4
    assert len(selected_configs) == 4
    assert {
        settings.model
        for selected_config in selected_configs
        for settings in selected_config.orchestration.agents.values()
    } == {"reasoning", "openrouter_test"}
    assert len(list((run_directory / "combinations").iterdir())) == 4


@pytest.mark.asyncio
async def test_evaluation_runner_calculates_all_quality_and_operational_metrics(
    repository_root: Path, tmp_path: Path
) -> None:
    dataset = tmp_path / "quality.yaml"
    dataset.write_text(
        """cases:
  - id: grounded_qa
    type: qa
    prompt: Explain DataPipeline
    expected_source_paths: [data_pipeline.v]
    expected_entities: [DataPipeline]
    expected_answer_entities: [DataPipeline]
  - id: generated_rtl
    type: code_generation
    prompt: Generate Demo
    expected_module: Demo
    expected_ports:
      - {name: clk, direction: input}
""",
        encoding="utf-8",
    )
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "datasets": [dataset],
                    "metrics": [
                        EvaluationMetric.CORRECTNESS,
                        EvaluationMetric.GROUNDING,
                        EvaluationMetric.RETRIEVAL,
                        EvaluationMetric.VALIDATION,
                    ],
                    "approaches": [ApproachType.MULTI_AGENT_GRAPHRAG],
                    "model_profiles": ["reasoning"],
                }
            ),
            "telemetry": config.telemetry.model_copy(
                update={"persistence_path": tmp_path / "runs"}
            ),
        }
    )

    run_directory = await EvaluationRunner(config, lambda _config: QualityStubService()).run()
    results = json.loads((run_directory / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))

    assert results[0]["correctness"]["scores"]["correctness"] == 1.0
    assert results[0]["execution_success"] is True
    assert results[0]["task_success"] is True
    assert results[0]["grounding"]["scores"]["grounding_accuracy"] == 1.0
    assert results[1]["validation"]["source"] == "answer"
    assert results[1]["validation"]["scores"]["validation_accuracy"] == 1.0
    assert metrics["correctness"]["evaluated_requests"] == 2
    assert metrics["grounding"]["evaluated_requests"] == 1
    assert metrics["retrieval"]["evaluated_requests"] == 1
    assert metrics["validation"]["evaluated_requests"] == 1
    assert metrics["execution_success_rate"] == 1.0
    assert metrics["task_success_rate"] == 1.0
    assert metrics["total_tokens"] == 45
    assert metrics["cached_input_tokens"] == 7
    assert metrics["cached_input_token_ratio"] == pytest.approx(7 / 30)
    assert metrics["average_total_tokens_per_request"] == 22.5
    assert metrics["average_tokens_per_llm_call"] == 22.5
    assert metrics["average_latest_context_tokens"] == 15.0
    assert metrics["average_peak_context_tokens"] == 15.0
    assert metrics["max_peak_context_tokens"] == 20
    assert metrics["average_history_messages"] == 1.0
    assert metrics["average_latency_seconds"] == 1.5
    assert metrics["p50_latency_seconds"] == 1.5
    assert metrics["p95_latency_seconds"] == pytest.approx(1.95)
