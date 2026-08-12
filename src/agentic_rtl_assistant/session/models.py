"""Conversation state is deliberately separate from per-request graph state."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(slots=True)
class ConversationState:
    session_id: str
    recent_messages: list[ConversationMessage] = field(default_factory=list)
    resolved_entities: list[str] = field(default_factory=list)
    compact_project_facts: list[str] = field(default_factory=list)
