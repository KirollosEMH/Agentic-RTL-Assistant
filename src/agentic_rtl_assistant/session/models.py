from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agentic_rtl_assistant.models.types import ModelMessage


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class ConversationState:
    session_id: str
    recent_messages: list[ConversationMessage] = field(default_factory=list)
    resolved_entities: list[str] = field(default_factory=list)
    compact_project_facts: list[str] = field(default_factory=list)

    def record_turn(self, user_text: str, assistant_text: str, *, limit: int) -> None:
        self.recent_messages.extend(
            (
                ConversationMessage("user", user_text),
                ConversationMessage("assistant", assistant_text),
            )
        )
        if limit == 0:
            self.recent_messages.clear()
        elif len(self.recent_messages) > limit:
            del self.recent_messages[:-limit]


def as_model_messages(
    messages: tuple[ConversationMessage, ...],
) -> tuple[ModelMessage, ...]:
    return tuple(ModelMessage(message.role, message.content) for message in messages)


def contextualize_request(
    request: str, messages: tuple[ConversationMessage, ...]
) -> str:
    if not messages:
        return request
    history = "\n".join(f"{message.role}: {message.content}" for message in messages)
    return f"Conversation:\n{history}\n\nCurrent request:\n{request}"
