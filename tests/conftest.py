from pathlib import Path

import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def rtl_root(repository_root: Path) -> Path:
    return repository_root / "descriptive_verilog_design"
