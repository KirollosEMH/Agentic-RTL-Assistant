from pathlib import Path

import pytest

from agentic_rtl_assistant.app.service import ApplicationService
from agentic_rtl_assistant.approaches.factory import ApproachFactory
from agentic_rtl_assistant.config import load_config
from agentic_rtl_assistant.telemetry.collector import TelemetryCollector
from tests.model_doubles import StubModelProviderFactory


@pytest.fixture
def service(repository_root: Path) -> ApplicationService:
    config_path = repository_root / "config" / "default.yaml"
    config = load_config(config_path, default_path=config_path, environment={})
    telemetry = TelemetryCollector(config.telemetry.enabled)
    model_factory = StubModelProviderFactory(config.models)
    approach_factory = ApproachFactory(config, telemetry, model_factory)
    return ApplicationService(config, approach_factory, telemetry)


@pytest.mark.asyncio
async def test_list_modules_vertical_slice(service: ApplicationService) -> None:
    result = await service.ask("What modules are implemented in the project?")

    assert result.succeeded
    assert result.answer is not None
    assert all(
        module in result.answer for module in ("DataPipeline", "CounterProducer", "DataConsumer")
    )
    assert {"DataPipeline", "CounterProducer", "DataConsumer"} <= set(result.evidence.entities)
    assert result.usage.llm_calls == 2


@pytest.mark.asyncio
async def test_pipeline_hierarchy_is_grounded(service: ApplicationService) -> None:
    result = await service.ask(
        "How is DataPipeline related to CounterProducer and DataConsumer?"
    )

    facts = {(item.source, item.relation, item.target) for item in result.evidence.relations}
    assert ("DataPipeline", "INSTANTIATES", "CounterProducer") in facts
    assert ("DataPipeline", "INSTANTIATES", "DataConsumer") in facts


@pytest.mark.asyncio
async def test_counterproducer_controls_retrieve_exact_source(service: ApplicationService) -> None:
    result = await service.ask("What signals control the behavior of CounterProducer?")

    source = "\n".join(item.content for item in result.evidence.source_evidence)
    assert "enable && ready_in" in source
    assert result.answer is not None and "ready_in" in result.answer


@pytest.mark.asyncio
async def test_fifo_generation_is_validated_without_writing(service: ApplicationService) -> None:
    result = await service.ask(
        "Implement a module named FifoBuffer that acts as a buffer "
        "between CounterProducer and DataConsumer."
    )

    assert result.generated_code is not None
    assert "module FifoBuffer" in result.generated_code
    assert result.validation is not None and result.validation.valid
    assert not (service.config.project.root / "fifo_buffer.v").exists()
