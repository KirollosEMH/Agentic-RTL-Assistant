from typing import Literal

from agentic_rtl_assistant.agents.types import Intent
from agentic_rtl_assistant.orchestration.state import AgentState


def route_after_classify(state: AgentState) -> Literal["graph_retrieve", "finalize"]:
    return "finalize" if state.get("intent") is Intent.UNKNOWN else "graph_retrieve"


def route_after_retrieval(state: AgentState) -> Literal["explain", "generate"]:
    if state.get("intent") in {Intent.GENERATE, Intent.MODIFY}:
        return "generate"
    return "explain"


def route_after_validation(
    state: AgentState, *, max_repair_attempts: int
) -> Literal["repair", "finalize"]:
    validation = state.get("validation_result")
    attempts = state.get("repair_attempts", 0)
    if validation is not None and not validation.valid and attempts < max_repair_attempts:
        return "repair"
    return "finalize"
