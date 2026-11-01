import shutil
from pathlib import Path
import pytest
import tiny_models

from router_ml import data

FIXTURES = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def mini(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    shutil.copy(FIXTURES / "mini.csv", raw / "train.csv")
    shutil.copy(FIXTURES / "mini.csv", raw / "test.csv")
    return data.build(raw, val_size=0.34)