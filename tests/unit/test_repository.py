from pathlib import Path

import pytest

from agentic_rtl_assistant.rtl.repository import PathConfinementError, RTLRepository


def test_verilog_discovery_and_read(rtl_root: Path) -> None:
    repository = RTLRepository(rtl_root)

    assert {path.name for path in repository.list_verilog_files()} == {
        "counter_producer.v",
        "data_consumer.v",
        "data_pipeline.v",
    }
    assert "module DataPipeline" in repository.read_source("data_pipeline.v")


def test_path_traversal_is_rejected(rtl_root: Path) -> None:
    repository = RTLRepository(rtl_root)

    with pytest.raises(PathConfinementError):
        repository.read_source("../pyproject.toml")
