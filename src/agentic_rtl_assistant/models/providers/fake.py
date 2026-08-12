"""Deterministic offline provider used by tests and the starter demo."""

from __future__ import annotations

import re

from agentic_rtl_assistant.models.types import ModelRequest, ModelResponse, TokenUsage


class FakeModelProvider:
    name = "fake"

    def __init__(self, scripted_responses: list[str] | None = None) -> None:
        self._responses = list(scripted_responses or [])

    async def generate(self, request: ModelRequest) -> ModelResponse:
        content = self._responses.pop(0) if self._responses else self._respond(request)
        input_tokens = sum(len(message.content.split()) for message in request.messages)
        return ModelResponse(
            content=content,
            provider=self.name,
            model=request.model,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=len(content.split()),
                cached_input_tokens=None,
                llm_calls=1,
            ),
        )

    @staticmethod
    def _respond(request: ModelRequest) -> str:
        system = request.messages[0].content.lower() if request.messages else ""
        user = request.messages[-1].content if request.messages else ""
        lowered = user.lower()
        if "classify the request" in system:
            words = set(re.findall(r"[a-z_]+", lowered))
            if words & {"implement", "generate", "create"}:
                return '{"intent": "generate"}'
            if words & {"modify", "change", "update"}:
                return '{"intent": "modify"}'
            if "what modules" in lowered or "list modules" in lowered:
                return '{"intent": "list"}'
            return '{"intent": "explain"}'
        if "generate only synthesizable" in system:
            return """```verilog
module FifoBuffer (
    input wire clk, input wire rst_n,
    input wire [7:0] data_in, input wire data_valid,
    output wire ready_in, output reg [7:0] data_out,
    output reg data_out_valid, input wire ready_out
);
    reg full;
    assign ready_in = !full || ready_out;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin full <= 1'b0; data_out <= 8'd0; data_out_valid <= 1'b0; end
        else begin
            if (ready_in) begin
                full <= data_valid;
                data_out_valid <= data_valid;
                if (data_valid) data_out <= data_in;
            end
        end
    end
endmodule
```"""
        if "repair the supplied verilog" in system:
            match = re.search(r"```verilog\s*(.*?)```", user, re.DOTALL)
            return f"```verilog\n{match.group(1).strip()}\n```" if match else user
        if "what modules" in lowered:
            return "The project implements DataPipeline, CounterProducer, and DataConsumer."
        if "related" in lowered or "relationship" in lowered:
            return (
                "DataPipeline is the top level and instantiates CounterProducer and DataConsumer; "
                "it wires the producer's data/valid outputs to the consumer and returns ready."
            )
        if "control" in lowered and "counterproducer" in lowered:
            return (
                "CounterProducer resets on active-low rst_n and produces a value on clk edges only "
                "when enable and ready_in are both high."
            )
        return "The supplied evidence does not establish a more specific answer."
