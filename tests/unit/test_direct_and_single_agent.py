from __future__ import annotations

import json
from pathlib import Path

from agentic_rtl_assistant.approaches.base import RunContext, UserRequest
from agentic_rtl_assistant.approaches.direct import DirectLLMApproach
from agentic_rtl_assistant.approaches.single_agent import SingleAgentApproach
from agentic_rtl_assistant.models.types import (
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from agentic_rtl_assistant.rtl.parser import PyVerilogParser
from agentic_rtl_assistant.rtl.repository import RTLRepository
from agentic_rtl_assistant.rtl.tools import RTLWriteTool, WriteRequest
from agentic_rtl_assistant.rtl.validator import RTLValidator
from agentic_rtl_assistant.session.models import ConversationMessage
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

    result = await approach.run(
        UserRequest.create("Describe the design."),
        RunContext(
            recent_messages=(
                ConversationMessage("user", "Earlier question"),
                ConversationMessage("assistant", "Earlier answer"),
            )
        ),
    )

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
    assert result.context_window.latest_input_tokens == 10
    assert result.context_window.peak_input_tokens == 10
    assert result.context_window.history_messages == 2


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
    assert result.context_window.latest_input_tokens == 10
    assert result.context_window.peak_input_tokens == 10
    assert result.context_window.history_messages == 0
    assert [event.event_type for event in result.traces].count(EventType.TOOL_STARTED) == 2
    assert [event.event_type for event in result.traces].count(EventType.TOOL_COMPLETED) == 2


async def test_single_agent_recovers_first_concatenated_json_action(
    rtl_root: Path,
) -> None:
    provider = ScriptedProvider(
        [
            'I will inspect it. {"tool":"read_file","arguments":'
            '{"path":"counter_producer.v"}}'
            '{"answer":"This ungrounded answer must not be accepted."}',
            '{"answer":"CounterProducer is defined in [counter_producer.v:7-31]."}',
        ]
    )
    approach = SingleAgentApproach(
        provider,
        RTLRepository(rtl_root),
        model="test-model",
        provider_name=provider.name,
        prompt="Use project tools.",
        telemetry=TelemetryCollector(),
    )

    result = await approach.run(UserRequest.create("Find CounterProducer."), RunContext())

    assert result.succeeded
    assert len(provider.requests) == 2
    assert result.answer == "CounterProducer is defined in [counter_producer.v:7-31]."
    assert provider.requests[1].messages[-2].content == (
        '{"tool": "read_file", "arguments": {"path": "counter_producer.v"}}'
    )
    assert "module CounterProducer" in provider.requests[1].messages[-1].content


async def test_single_agent_does_not_accept_malformed_text_as_final_answer(
    rtl_root: Path,
) -> None:
    provider = ScriptedProvider(
        [
            '{"tool":"read_file","arguments":{"path":"data_pipeline.v"}}',
            "This is not a valid JSON answer.",
            '{"answer":"DataPipeline is defined in [data_pipeline.v:4-34]."}',
        ]
    )
    approach = SingleAgentApproach(
        provider,
        RTLRepository(rtl_root),
        model="test-model",
        provider_name=provider.name,
        prompt="Use project tools.",
        telemetry=TelemetryCollector(),
    )

    result = await approach.run(UserRequest.create("Find DataPipeline."), RunContext())

    assert result.succeeded
    assert len(provider.requests) == 3
    assert result.answer == "DataPipeline is defined in [data_pipeline.v:4-34]."
    assert "Return one JSON object" in provider.requests[2].messages[-1].content


async def test_single_agent_returns_verilog_after_recovered_tool_actions(
    rtl_root: Path,
) -> None:
    verilog = """```verilog
module FifoBuffer(input wire clk);
endmodule
```"""
    provider = ScriptedProvider(
        [
            '{"tool":"read_file","arguments":{"path":"counter_producer.v"}}'
            '{"tool":"read_file","arguments":{"path":"data_consumer.v"}}',
            '{"tool":"read_file","arguments":{"path":"data_consumer.v"}}',
            json.dumps({"answer": verilog}),
        ]
    )
    approach = SingleAgentApproach(
        provider,
        RTLRepository(rtl_root),
        model="test-model",
        provider_name=provider.name,
        prompt="Use project tools.",
        telemetry=TelemetryCollector(),
    )

    result = await approach.run(UserRequest.create("Implement FifoBuffer."), RunContext())

    assert result.succeeded
    assert result.answer == verilog
    assert [item.path for item in result.evidence.source_evidence] == [
        "counter_producer.v",
        "data_consumer.v",
    ]


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


async def test_single_agent_can_write_approved_rtl_file(tmp_path: Path) -> None:
    rtl_root = tmp_path / "rtl"
    rtl_root.mkdir()
    (rtl_root / "existing.v").write_text(
        "module Existing; endmodule\n", encoding="utf-8"
    )
    repository = RTLRepository(rtl_root, allow_writes=True)
    provider = ScriptedProvider(
        [
            '{"tool":"read_file","arguments":{"path":"existing.v"}}',
            '{"tool":"write_file","arguments":{"path":"new_module.v",'
            '"content":"module NewModule; endmodule"}}',
            '{"answer":"Created new_module.v after inspecting existing.v."}',
        ]
    )
    write_tool = RTLWriteTool(
        repository,
        RTLValidator(PyVerilogParser(repository)),
        require_confirmation=True,
    )
    confirmations: list[WriteRequest] = []

    async def approve(request: WriteRequest) -> bool:
        confirmations.append(request)
        return True

    approach = SingleAgentApproach(
        provider,
        repository,
        model="test-model",
        provider_name=provider.name,
        prompt="Use project tools.",
        telemetry=TelemetryCollector(),
        write_tool=write_tool,
    )

    result = await approach.run(
        UserRequest.create("Inspect the project and create NewModule."),
        RunContext(write_confirmation=approve),
    )

    assert result.succeeded
    assert result.written_files == ("new_module.v",)
    assert confirmations[0].path == "new_module.v"
    assert (rtl_root / "new_module.v").read_text(encoding="utf-8") == (
        "module NewModule; endmodule\n"
    )
