from pathlib import Path

from pyverilog.vparser.parser import parse

HERE = Path(__file__).resolve().parent

verilog_files = [
    str(HERE / "data_pipeline.v"),
    str(HERE / "counter_producer.v"),
    str(HERE / "data_consumer.v"),
]

ast, directives = parse(verilog_files)

print("Parse successful.")
ast.show()
