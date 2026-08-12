from agentic_rtl_assistant.orchestration.routing import route_after_validation
from agentic_rtl_assistant.rtl.types import ValidationResult


def test_repair_attempt_limit_is_bounded() -> None:
    state = {
        "validation_result": ValidationResult(False, ("syntax",), ("parser",)),
        "repair_attempts": 2,
    }

    assert route_after_validation(state, max_repair_attempts=2) == "finalize"


def test_failed_first_pass_routes_to_repair() -> None:
    state = {
        "validation_result": ValidationResult(False, ("syntax",), ("parser",)),
        "repair_attempts": 0,
    }

    assert route_after_validation(state, max_repair_attempts=2) == "repair"
