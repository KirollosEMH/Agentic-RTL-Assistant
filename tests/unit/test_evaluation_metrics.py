from pathlib import Path

import pytest

from agentic_rtl_assistant.evaluation.correctness import score_correctness
from agentic_rtl_assistant.evaluation.grounding import score_grounding
from agentic_rtl_assistant.evaluation.validation import (
    ExpectedPort,
    extract_verilog,
    score_validation,
)
from agentic_rtl_assistant.knowledge.evidence import EvidencePack, SourceEvidence
from agentic_rtl_assistant.rtl.parser import PyVerilogParser
from agentic_rtl_assistant.rtl.types import ValidationResult


def test_correctness_scores_entities_and_required_fact_terms() -> None:
    scores = score_correctness(
        "DataPipeline contains instances of DataConsumer and CounterProducer.",
        expected_answer_entities=["DataPipeline", "CounterProducer", "MissingModule"],
        expected_answer_facts=[
            "DataPipeline instances CounterProducer",
            "DataPipeline instantiates MissingModule",
        ],
    )

    assert scores.answer_entity_recall == pytest.approx(2 / 3)
    assert scores.answer_fact_recall == 0.5
    assert scores.correctness == pytest.approx((2 / 3 + 0.5) / 2)


def test_grounding_validates_citations_against_project_and_evidence(tmp_path: Path) -> None:
    source = tmp_path / "pipeline.v"
    source.write_text("module DataPipeline;\nwire ready;\nendmodule\n", encoding="utf-8")
    evidence = EvidencePack(
        entities=("DataPipeline",),
        source_evidence=(
            SourceEvidence("pipeline.v", 1, 3, source.read_text(encoding="utf-8"), "test"),
        ),
    )

    scores = score_grounding(
        "DataPipeline is defined here [pipeline.v:1-3]. A bad citation is [missing.v:1].",
        evidence,
        project_root=tmp_path,
        expected_source_paths=["pipeline.v"],
        expected_answer_entities=["DataPipeline"],
    )

    assert scores.citation_precision == 0.5
    assert scores.source_citation_recall == 1.0
    assert scores.answer_entity_support == 1.0
    assert scores.grounding_accuracy == pytest.approx(5 / 6)


def test_validation_scores_parser_module_and_port_contract() -> None:
    source = """
module FifoBuffer (
    input wire clk,
    input wire [7:0] data_in,
    output wire [7:0] data_out
);
assign data_out = data_in;
endmodule
"""

    scores = score_validation(
        source,
        parser=PyVerilogParser(),
        runtime_validation=ValidationResult(True, (), ("parser",)),
        expected_module="FifoBuffer",
        expected_ports=[
            ExpectedPort("clk", "input"),
            ExpectedPort("data_in", "input", "[7:0]"),
            ExpectedPort("data_out", "output", "[7:0]"),
        ],
    )

    assert scores.validation_accuracy == 1.0
    assert scores.parser_valid == 1.0
    assert scores.runtime_validation_pass == 1.0
    assert scores.expected_module_match == 1.0
    assert scores.expected_port_recall == 1.0
    assert scores.expected_port_direction_accuracy == 1.0
    assert scores.expected_port_width_accuracy == 1.0


def test_validation_marks_missing_code_as_failed() -> None:
    scores = score_validation(
        None,
        parser=PyVerilogParser(),
        expected_module="FifoBuffer",
        expected_ports=[ExpectedPort("clk", "input")],
    )

    assert scores.parser_valid == 0.0
    assert scores.validation_accuracy == 0.0


def test_extract_verilog_from_markdown_answer() -> None:
    answer = "Here is the implementation:\n```verilog\nmodule Demo; endmodule\n```"

    assert extract_verilog(answer) == "module Demo; endmodule"
