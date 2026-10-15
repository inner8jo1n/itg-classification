"""Loading and validation of the raw Kaggle dataset."""

from pathlib import Path

import pandas as pd

from itg_classification.config import LABELS, RAW_COLUMNS, RAW_DATA_PATH


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Reads the raw CSV as a frame with `text` and `label` columns.

    Fails early if a column is missing or a label is unknown, so that a
    broken file never reaches training.
    """
    frame = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} has no columns {sorted(missing)}")
    frame = frame[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    unknown = set(frame["label"].dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"{path} has unknown labels {sorted(unknown)}")
    return frame
