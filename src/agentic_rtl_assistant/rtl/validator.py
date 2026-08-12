"""Deterministic validation abstraction with an initial parser layer."""

from agentic_rtl_assistant.rtl.parser import RTLParseError, RTLParser
from agentic_rtl_assistant.rtl.types import ValidationResult


class RTLValidator:
    def __init__(self, parser: RTLParser) -> None:
        self.parser = parser

    def validate_verilog(
        self, source: str, *, expected_module: str | None = None
    ) -> ValidationResult:
        try:
            modules = self.parser.parse_text(source)
        except RTLParseError as exc:
            return ValidationResult(False, (str(exc),), ("parser",))
        if expected_module and expected_module not in {module.name for module in modules}:
            return ValidationResult(
                False,
                (f"expected module not found: {expected_module}",),
                ("parser", "expected_module"),
            )
        return ValidationResult(True, (), ("parser",))
