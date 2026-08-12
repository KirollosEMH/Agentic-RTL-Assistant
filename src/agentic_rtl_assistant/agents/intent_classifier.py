import json

from agentic_rtl_assistant.agents.base import Agent
from agentic_rtl_assistant.agents.types import AgentResult, Intent, IntentInput, IntentOutput
from agentic_rtl_assistant.models.types import ModelMessage, ModelRequest


class IntentClassifierAgent(Agent[IntentInput, AgentResult[IntentOutput]]):
    name = "intent_classifier"

    async def execute(self, context: IntentInput) -> AgentResult[IntentOutput]:
        response = await self.model_client.generate(
            ModelRequest(
                model=self.profile.model,
                messages=(
                    ModelMessage("system", self.prompt),
                    ModelMessage("user", context.user_request),
                ),
                temperature=self.config.temperature,
                max_output_tokens=self.profile.max_output_tokens,
            )
        )
        try:
            intent = Intent(json.loads(response.content)["intent"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"intent classifier returned invalid JSON: {response.content}"
            ) from exc
        return AgentResult(output=IntentOutput(intent), usage=response.usage)
