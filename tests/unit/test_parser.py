from pathlib import Path

from agentic_rtl_assistant.rtl.parser import PyVerilogParser
from agentic_rtl_assistant.rtl.repository import RTLRepository


def test_pyverilog_adapter_extracts_compact_module_structure(rtl_root: Path) -> None:
    repository = RTLRepository(rtl_root)
    modules = PyVerilogParser(repository).parse_project()
    by_name = {module.name: module for module in modules}

    assert set(by_name) == {"DataPipeline", "CounterProducer", "DataConsumer"}
    pipeline_instances = {
        (instance.name, instance.module) for instance in by_name["DataPipeline"].instances
    }
    assert pipeline_instances == {
        ("u_counter_producer", "CounterProducer"),
        ("u_data_consumer", "DataConsumer"),
    }
    producer_ports = {port.name for port in by_name["CounterProducer"].ports}
    assert {"clk", "rst_n", "enable", "ready_in", "data_out", "data_valid"} <= producer_ports
