import pandas as pd
import pytest

from itg_classification.config import LABELS, RAW_DATA_PATH
from itg_classification.load_data import load_dataset


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_dataset_renames_columns(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        {"Sentence": ["Shares rose"], "Sentiment": ["positive"]},
    )

    frame = load_dataset(path)

    assert frame.columns.tolist() == ["text", "label"]
    assert frame.iloc[0].tolist() == ["Shares rose", "positive"]


def test_load_dataset_rejects_missing_column(tmp_path):
    path = write_csv(tmp_path / "data.csv", {"Sentence": ["Shares rose"]})

    with pytest.raises(ValueError, match="Sentiment"):
        load_dataset(path)


def test_load_dataset_rejects_unknown_label(tmp_path):
    path = write_csv(
        tmp_path / "data.csv",
        {"Sentence": ["Shares rose"], "Sentiment": ["bullish"]},
    )

    with pytest.raises(ValueError, match="bullish"):
        load_dataset(path)


def test_raw_dataset_loads():
    frame = load_dataset(RAW_DATA_PATH)

    assert len(frame) > 0
    assert set(frame["label"]) == set(LABELS)
