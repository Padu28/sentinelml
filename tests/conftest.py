import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def settings():
    from sentinelml.common.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def stream():
    from sentinelml.ingestion.synthetic import StreamSpec, generate_stream

    return generate_stream(StreamSpec(n_records=6000, drift_start=3000, batch_size=250,
                                      quality_fault_batches=[20], drift_type="sudden", seed=1))


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    from sentinelml.storage.database import Database

    return Database(tmp_path / "test.db")
