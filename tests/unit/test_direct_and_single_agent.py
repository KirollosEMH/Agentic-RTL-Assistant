from __future__ import annotations

from pathlib import Path

from agentic_rtl_assistant.approaches.base import RunContext, UserRequest
from agentic_rtl_assistant.approaches.direct import DirectLLMApproach
from agentic_rtl_assistant.approaches.single_agent import SingleAgentApproach
from agentic_rtl_assistant.models.types import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from agentic_rtl_assistant.telemetry.traces import EventType


class ScriptedProvider:
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=next(self.responses),
            provider=self.name,
            model=request.model,
            usage=TokenUsage(10, 2, 0, 1),
        )


async def test_direct_llm_receives_every_project_rtl_file(rtl_root: Path) -> None:
    provider = ScriptedProvider(["A grounded answer."])
    approach = DirectLLMApproach(
        provider,
        model="test-model",
        provider_name=provider.name,
        prompt="Use the supplied project source.",
        repository=RTLRepository(rtl_root),
        telemetry=TelemetryCollector(),
    )

    result = await approach.run(UserRequest.create("Describe the design."), RunContext())

    assert len(provider.requests) == 1
    project_prompt = provider.requests[0].messages[-1].content
    expected_paths = {
        "counter_producer.v",
        "data_consumer.v",
        "data_pipeline.v",
    }
    for path in expected_paths:
        assert f"[{path}:1-" in project_prompt
        assert (rtl_root / path).read_text(encoding="utf-8") in project_prompt
    assert "parse_with_pyverilog.py" not in project_prompt
    assert {item.path for item in result.evidence.source_evidence} == expected_paths
    assert result.evidence.metrics.source_chunks_retrieved == 3
    assert result.usage.llm_calls == 1


async def test_single_agent_lists_and_reads_project_files(rtl_root: Path) -> None:
    provider = ScriptedProvider(
        [
            '{"tool":"list_files","arguments":{}}',
            '{"tool":"read_file","arguments":{"path":"counter_producer.v"}}',
            '{"answer":"CounterProducer increments when enable and ready_in are high '
            '[counter_producer.v:19-23]."}',
        ]
    )
    telemetry = TelemetryCollector()
    approach = SingleAgentApproach(
        provider,
        RTLRepository(rtl_root),
        model="test-model",
        provider_name=provider.name,
        prompt="Use project tools.",
        telemetry=telemetry,
    )

    result = await approach.run(UserRequest.create("When does it increment?"), RunContext())

    assert result.succeeded
    assert len(provider.requests) == 3
    assert '"counter_producer.v"' in provider.requests[1].messages[-1].content
    assert "module CounterProducer" in provider.requests[2].messages[-1].content
    assert [item.path for item in result.evidence.source_evidence] == [
        "counter_producer.v"
    ]
    assert result.usage == TokenUsage(30, 6, 0, 3)
    assert [event.event_type for event in result.traces].count(EventType.TOOL_STARTED) == 2
    assert [event.event_type for event in result.traces].count(EventType.TOOL_COMPLETED) == 2


async def test_single_agent_rejects_paths_outside_project(rtl_root: Path) -> None:
    provider = ScriptedProvider(
        [
            '{"tool":"read_file","arguments":{"path":"../pyproject.toml"}}',
            '{"answer":"I read it."}',
            '{"tool":"list_files","arguments":{}}',
            '{"tool":"read_file","arguments":{"path":"data_pipeline.v",'
            '"start_line":1,"end_line":3}}',
            '{"answer":"The project contains three RTL files."}',
        ]
    )
    approach = SingleAgentApproach(
        provider,
        RTLRepository(rtl_root),
        model="test-model",
        provider_name=provider.name,
        prompt="Use project tools.",
        telemetry=TelemetryCollector(),
        max_steps=5,
    )

    result = await approach.run(UserRequest.create("Inspect the project."), RunContext())

    assert result.succeeded
    assert len(provider.requests) == 5
    assert "path escapes project root" in provider.requests[1].messages[-1].content
    assert result.answer == "The project contains three RTL files."
    assert [item.path for item in result.evidence.source_evidence] == ["data_pipeline.v"]
