from agentic_rtl_assistant.agents.base import Agent
from agentic_rtl_assistant.agents.rtl_codegen import extract_verilog
from agentic_rtl_assistant.agents.types import AgentResult, CodeGenerationOutput, RepairInput
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest
from agentic_rtl_assistant.session.models import as_model_messages


class RTLRepairAgent(Agent[RepairInput, AgentResult[CodeGenerationOutput]]):
    name = "rtl_repair"

    async def execute(self, context: RepairInput) -> AgentResult[CodeGenerationOutput]:
        errors = "\n".join(f"- {error}" for error in context.validation_errors)
        response = await self.model_client.generate(
            ModelRequest(
                model=self.profile.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    *as_model_messages(context.recent_messages),
                    ModelMessage(
                        "user",
                        f"Requirement:\n{context.requirement}\n\nValidation errors:\n{errors}"
                        f"\n\nGenerated RTL:\n```verilog\n{context.generated_code}\n```"
                        f"\n\nEvidence:\n{context.evidence.to_prompt()}",
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
