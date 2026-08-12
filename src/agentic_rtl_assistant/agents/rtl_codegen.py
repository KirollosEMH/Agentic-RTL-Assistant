import re

from agentic_rtl_assistant.agents.base import Agent
from agentic_rtl_assistant.agents.types import (
    AgentResult,
    CodeGenerationInput,
    CodeGenerationOutput,
)
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest
from agentic_rtl_assistant.session.models import as_model_messages


def extract_verilog(content: str) -> str:
    match = re.search(r"```(?:system)?verilog\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else content).strip()


class RTLCodeAgent(Agent[CodeGenerationInput, AgentResult[CodeGenerationOutput]]):
    name = "rtl_codegen"

    async def execute(self, context: CodeGenerationInput) -> AgentResult[CodeGenerationOutput]:
        response = await self.model_client.generate(
            ModelRequest(
                model=self.profile.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    *as_model_messages(context.recent_messages),
                    ModelMessage(
                        "user",
                        f"Requirement:\n{context.requirement}\n\nEvidence:\n{context.evidence.to_prompt()}",
                    ),
                ),
                temperature=self.config.temperature,
                max_output_tokens=self.profile.max_output_tokens,
                metadata={"session_id": context.session_id} if context.session_id else {},
            )
        )
        return AgentResult(
            output=CodeGenerationOutput(extract_verilog(response.content)), usage=response.usage
        )
