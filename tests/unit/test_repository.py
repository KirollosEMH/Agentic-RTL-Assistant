from pathlib import Path

import pytest

from agentic_rtl_assistant.rtl.parser import PyVerilogParser
from agentic_rtl_assistant.rtl.repository import (
    PathConfinementError,
    RTLRepository,
    RTLRepositoryError,
)
from agentic_rtl_assistant.rtl.tools import RTLWriteTool, WriteDeclinedError, WriteRequest
from agentic_rtl_assistant.rtl.validator import RTLValidator


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


@pytest.mark.asyncio
async def test_write_tool_validates_confirms_and_confines_writes(tmp_path: Path) -> None:
    rtl_root = tmp_path / "rtl"
    rtl_root.mkdir()
    repository = RTLRepository(rtl_root, allow_writes=True)
    tool = RTLWriteTool(
        repository,
        RTLValidator(PyVerilogParser(repository)),
        require_confirmation=True,
    )
    confirmations: list[WriteRequest] = []

    async def approve(request: WriteRequest) -> bool:
        confirmations.append(request)
        return True

    result = await tool.execute(
        WriteRequest("fifo_buffer.v", "module FifoBuffer; endmodule"), approve
    )

    assert result.path == "fifo_buffer.v"
    assert confirmations[0].path == "fifo_buffer.v"
    assert (rtl_root / "fifo_buffer.v").read_text(encoding="utf-8") == (
        "module FifoBuffer; endmodule\n"
    )

    with pytest.raises(RTLRepositoryError, match="already exists"):
        await tool.execute(
            WriteRequest("fifo_buffer.v", "module Replacement; endmodule"), approve
        )
    await tool.execute(
        WriteRequest(
            "fifo_buffer.v",
            "module Replacement; endmodule",
            overwrite=True,
        ),
        approve,
    )
    assert (rtl_root / "fifo_buffer.v").read_text(encoding="utf-8") == (
        "module Replacement; endmodule\n"
    )

    async def decline(_request: WriteRequest) -> bool:
        return False

    with pytest.raises(WriteDeclinedError, match="declined"):
        await tool.execute(
            WriteRequest("other.v", "module Other; endmodule"), decline
        )
    assert not (rtl_root / "other.v").exists()

    with pytest.raises(PathConfinementError):
        await tool.execute(
            WriteRequest("../escape.v", "module Escape; endmodule"), approve
        )
    with pytest.raises(RTLRepositoryError, match="RTL extension"):
        await tool.execute(
            WriteRequest("not_rtl.txt", "module InvalidPath; endmodule"), approve
        )
    with pytest.raises(ValueError, match="invalid RTL"):
        await tool.execute(WriteRequest("broken.v", "not verilog"), approve)
