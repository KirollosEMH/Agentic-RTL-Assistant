from agentic_rtl_assistant.agents.base import Agent
from agentic_rtl_assistant.agents.types import AgentResult, ExplanationInput, ExplanationOutput
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest
from agentic_rtl_assistant.session.models import as_model_messages


class RTLExplanationAgent(Agent[ExplanationInput, AgentResult[ExplanationOutput]]):
    name = "rtl_explanation"

    async def execute(self, context: ExplanationInput) -> AgentResult[ExplanationOutput]:
        response = await self.model_client.generate(
            ModelRequest(
                model=self.profile.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    *as_model_messages(context.recent_messages),
                    ModelMessage(
                        "user",
                        f"Question:\n{context.user_request}\n\nEvidence:\n{context.evidence.to_prompt()}",
                    ),
                ),
                temperature=self.config.temperature,
                max_output_tokens=self.profile.max_output_tokens,
                metadata={"session_id": context.session_id} if context.session_id else {},
            )
        )
        return AgentResult(output=ExplanationOutput(response.content.strip()), usage=response.usage)
