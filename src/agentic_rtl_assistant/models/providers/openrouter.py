from agentic_rtl_assistant.models.providers.openai_compatible import OpenAICompatibleProvider
from agentic_rtl_assistant.models.types import ModelRequest


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def _extra_body(self, request: ModelRequest) -> dict[str, object] | None:
        session_id = request.metadata.get("session_id")
        return {"session_id": session_id} if session_id else None
