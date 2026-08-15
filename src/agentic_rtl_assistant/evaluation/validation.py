from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import fmean

from agentic_rtl_assistant.rtl.parser import RTLParseError, RTLParser
from agentic_rtl_assistant.rtl.types import ModuleInfo, ValidationResult


@dataclass(frozen=True, slots=True)
class ExpectedPort:
    name: str
    direction: str
    width: str | None = None


@dataclass(frozen=True, slots=True)
class ExpectedInstance:
    module: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationScores:
    validation_accuracy: float
    parser_valid: float
    runtime_validation_pass: float | None
    expected_module_match: float | None
    expected_port_recall: float | None
    expected_port_direction_accuracy: float | None
    expected_port_width_accuracy: float | None
    expected_instance_recall: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def extract_verilog(answer: str | None) -> str | None:
    """Extract a Verilog candidate from a general model answer."""

    if not answer:
        return None
    fenced = re.search(
        r"```(?:systemverilog|verilog|sv|v)?\s*(.*?)```", answer, re.IGNORECASE | re.DOTALL
    )
    if fenced and "module" in fenced.group(1).casefold():
        return fenced.group(1).strip()
    module = re.search(r"\bmodule\b.*?\bendmodule\b", answer, re.IGNORECASE | re.DOTALL)
    return module.group(0).strip() if module else None


def _normalized_width(width: str | None) -> str | None:
    return re.sub(r"\s+", "", width) if width is not None else None


def _target_module(
    modules: tuple[ModuleInfo, ...], expected_module: str | None
) -> ModuleInfo | None:
    if expected_module:
        return next((module for module in modules if module.name == expected_module), None)
    return modules[0] if len(modules) == 1 else None


def score_validation(
    source: str | None,
    *,
    parser: RTLParser,
    runtime_validation: ValidationResult | None = None,
    expected_module: str | None = None,
    expected_ports: list[ExpectedPort] | tuple[ExpectedPort, ...] = (),
    expected_instances: list[ExpectedInstance] | tuple[ExpectedInstance, ...] = (),
) -> ValidationScores:
    modules: tuple[ModuleInfo, ...] = ()
    parser_valid = 0.0
    if source:
        try:
            modules = parser.parse_text(source)
            parser_valid = float(bool(modules))
        except RTLParseError:
            pass

    module_match = None
    if expected_module:
        module_match = float(any(module.name == expected_module for module in modules))

    port_recall = None
    direction_accuracy = None
    width_accuracy = None
    instance_recall = None
    target = _target_module(modules, expected_module)
    if expected_ports:
        actual = {port.name: port for port in target.ports} if target is not None else {}
        port_recall = sum(port.name in actual for port in expected_ports) / len(expected_ports)
        direction_accuracy = sum(
            port.name in actual
            and actual[port.name].direction.casefold() == port.direction.casefold()
            for port in expected_ports
        ) / len(expected_ports)
        ports_with_width = [port for port in expected_ports if port.width is not None]
        if ports_with_width:
            width_accuracy = sum(
                port.name in actual
                and _normalized_width(actual[port.name].width) == _normalized_width(port.width)
                for port in ports_with_width
            ) / len(ports_with_width)

    if expected_instances:
        actual_instances = target.instances if target is not None else ()
        instance_recall = sum(
            any(
                instance.module == expected.module
                and (expected.name is None or instance.name == expected.name)
                for instance in actual_instances
            )
            for expected in expected_instances
        ) / len(expected_instances)

    structural_components = [
        score
        for score in (
            parser_valid,
            float(runtime_validation.valid) if runtime_validation is not None else None,
            module_match,
            port_recall,
            direction_accuracy,
            width_accuracy,
            instance_recall,
        )
        if score is not None
    ]
    return ValidationScores(
        validation_accuracy=fmean(structural_components),
        parser_valid=parser_valid,
        runtime_validation_pass=(
            float(runtime_validation.valid) if runtime_validation is not None else None
        ),
        expected_module_match=module_match,
        expected_port_recall=port_recall,
        expected_port_direction_accuracy=direction_accuracy,
        expected_port_width_accuracy=width_accuracy,
        expected_instance_recall=instance_recall,
    )


def aggregate_validation_scores(
    scores: list[ValidationScores],
) -> dict[str, float | int | None]:
    names = tuple(ValidationScores.__dataclass_fields__)
    aggregated: dict[str, float | int | None] = {
        "evaluated_requests": len(scores),
        "runtime_validation_evaluated_requests": sum(
            score.runtime_validation_pass is not None for score in scores
        ),
    }
    for name in names:
        values = [value for score in scores if (value := getattr(score, name)) is not None]
        aggregated[name] = fmean(values) if values else None
    return aggregated
