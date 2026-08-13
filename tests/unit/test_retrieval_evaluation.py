import json
from pathlib import Path

import pytest

from agentic_rtl_assistant.approaches.base import RunResult
from agentic_rtl_assistant.config import load_config
from agentic_rtl_assistant.evaluation.retrieval import (
    aggregate_retrieval_scores,
    score_retrieval,
)
from agentic_rtl_assistant.evaluation.runner import EvaluationRunner
from agentic_rtl_assistant.knowledge.evidence import (
    EvidencePack,
    GraphRelation,
    SourceEvidence,
)


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
                update={"datasets": [dataset], "metrics": ["retrieval"]}
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
