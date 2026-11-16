import numpy as np
import pandas as pd
import pytest

from src.app.config import get_settings
from src.app.services import data_io
from src.app.services.data_io import DatasetError, json_safe, load_dataframe


def test_load_dataframe_caches_and_invalidates_on_change(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("x,y\n1,2\n3,4\n")
    first = load_dataframe(path)
    assert load_dataframe(path) is first  # cached
    path.write_text("x,y\n1,2\n3,4\n5,6\n")
    assert len(load_dataframe(path)) == 3  # changed file -> reloaded


def test_load_dataframe_errors(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        load_dataframe(tmp_path / "missing.csv")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(DatasetError):
        load_dataframe(empty)
    monkeypatch.setenv("MAX_DATASET_ROWS", "2")
    get_settings.cache_clear()
    big = tmp_path / "big.csv"
    big.write_text("a\n1\n2\n3\n")
    with pytest.raises(DatasetError, match="limit"):
        load_dataframe(big)
    get_settings.cache_clear()
    data_io.clear_cache()


def test_json_safe_handles_numpy_nan_and_timestamps():
    data = {"a": np.int64(3), "b": np.float64("nan"), "c": pd.Timestamp("2024-01-02"),
            "d": [np.float32(1.5), np.inf], "e": np.bool_(True)}
    assert json_safe(data) == {"a": 3, "b": None, "c": "2024-01-02T00:00:00",
                               "d": [1.5, None], "e": True}


def test_sha256_is_stable(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert data_io.sha256_file(p) == data_io.sha256_file(p)
    assert len(data_io.sha256_file(p)) == 64
