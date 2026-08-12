from agentic_rtl_assistant.session.models import ConversationState


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str) -> ConversationState:
        return self._sessions.setdefault(session_id, ConversationState(session_id))
